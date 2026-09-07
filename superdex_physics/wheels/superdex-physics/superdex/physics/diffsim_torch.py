# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch autograd bridge for differentiable SuperDex rollouts.

Two entry points: :class:`TorchRollout` for open-loop inputs (control sequences,
forces, parameters as tensors) and :class:`PolicyRollout` for a closed loop (a
torch policy maps the observed state to the controller targets and/or the
external forces - joint torques - at every step, and the policy parameters
receive the loss gradient, feedback path included).

:class:`TorchRollout` wraps a fixed scene + rollout schedule + loss into a
callable that maps input tensors to a scalar ``torch.Tensor`` loss, with the
backward pass served by the engine's discrete adjoint
(:class:`superdex.physics.diffsim_rollout.DifferentiableRollout`). That makes
the simulator a node in a torch autograd graph: upstream networks (policies
producing controls, parameter encoders, ...) receive exact simulation
gradients, and torch optimizers drive them.

Differentiable inputs (each group is opt-in at construction):

- ``controls`` - per-step pose-controller targets for articulated actors,
  shape ``(num_steps, total_control_dofs)``;
- ``forces`` - per-step external forces, shape
  ``(num_steps, total_force_dofs)`` (all six world-frame DoFs of a standalone
  rigid actor; the single-DoF joints of an articulated one);
- ``gravity`` - the scene gravity vector, shape ``(3,)``;
- ``contact_params`` - per-actor contact material parameters, shape
  ``(num_contact_actors, 4)`` in the engine's gradient order
  :data:`CONTACT_PARAM_FIELDS`;
- ``densities`` - per-actor mass densities, shape ``(num_density_actors,)``;
- ``initial_states`` - the initial nodal state of soft actors, shape
  ``(total_initial_state_size,)``: per declared actor its local-frame nodal
  displacements followed by its nodal velocities (both ``num_dofs`` long).
  Soft actors only for now: their state is a plain vector space, whereas the
  rigid/articulated initial state lives on a manifold (quaternion chart); the
  driver's ``RolloutResult`` still carries those gradients;
- ``soft_materials`` - per soft actor its material parameters, shape
  ``(num_soft_material_actors, 4)`` in the engine's gradient order
  :data:`SOFT_MATERIAL_FIELDS` (Young's modulus, Poisson's ratio, density,
  mass-damping coefficient). Homogeneous Lame-type materials only (the
  engine rejects other models and per-element fields at read time). The
  mass-damping coefficient is gated at zero in the engine, so its gradient
  at zero is the right-sided derivative.

Soft actors take part in the ``contact_params``, ``initial_states`` and
``soft_materials`` groups and in the gravity gradient; they carry no external
forces, no controller and no rigid-body density (their density is part of
``soft_materials``), so declaring one in the force, control or density groups
is an error.

Design and contract:

- The forward and backward sweeps both run inside ``forward`` (the adjoint
  needs the step states while they are captured), so gradients are always
  computed; ``backward`` only scales the stashed gradients by the incoming
  ``grad_output``. Calling the bridge is therefore "loss + gradients", not
  "loss only".
- Tensors must be CPU ``float64`` with the exact documented shape - anything
  else raises. The engine simulates in double precision
  (``SUPERDEX_PRECISION=double``, the ``superdex-physics-fp64`` package); the
  bridges refuse the single-precision engine at construction, and mixed
  precision would silently degrade the gradients, so it is rejected rather
  than converted.
- A group declared at construction requires its tensor at every call, and a
  tensor for an undeclared group is rejected: there are no implicit defaults.
- Every actor of a group belongs to the bridge's scene and appears once in
  that group (identity by handle; names only label the gradient results): an
  actor of another scene, even with the same name, and a repeated actor are
  rejected at construction. One actor may sit in several different groups.
- ``step_losses(step)`` is called once per step, during the forward rollout;
  the sweep differentiates the instances it returned (a factory may sample or
  consume data and still defines one objective).
- Losses follow the ``diffsim_rollout`` protocol (``value()`` and
  ``accumulate_output_grad()``); they are part of the bridge, not tensors, so
  the loss shape itself is fixed at construction.
- Every call restores the initial scene state captured at construction,
  applies the parameter tensors, and runs the full rollout: calls are
  independent and deterministic (single-threaded engine), which is what
  ``torch.autograd.gradcheck`` verifies against finite differences.
- The forward Newton tolerance bounds gradient fidelity (the adjoint assumes
  the step equations hold exactly). Configure the scene's non-linear solver
  tightly (e.g. ``abs_tol = 1e-12``) before building the bridge; with the
  default ``1e-3``-ish tolerances, friction-heavy gradients can be off by
  several percent against finite differences.

Every input group, torques included, passes ``torch.autograd.gradcheck``.
The torque gradients of a standalone rigid actor need the adjoint operator to
carry the moving-chart term of the external torque (the step residual is
evaluated in the chart of the iterate itself, so the step Jacobian is the
fixed-chart Hessian minus 1/2 [tau]x on the rotation block; the engine adds
its transpose in the adjoint solve since 2026-09-05). Before that the torque
and rotational gradients were off by half the rotation the torque induces in
a step (2.2e-4 relative at 0.3 N*m, 8.6e-4 at 1.2 N*m on a free cube); now
below 1e-8 at both, 1.4e-6 over five steps (``TorqueGradientTest``; finite
differences at eps 1e-4 - at eps 1e-6 the quotients carry the forward Newton's
stopping noise, 3e-5 at 1.2 N*m).

Example::

    tr = TorchRollout(
        scene, dt=0.01, num_steps=30,
        control_actors=[chain],
        terminal_losses=[my_loss],
    )
    controls = torch.zeros(30, tr.control_size, dtype=torch.float64,
                           requires_grad=True)
    loss = tr(controls=controls)
    loss.backward()          # controls.grad now holds dL/dcontrols
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

import numpy as np

import superdex.physics as physics
from superdex.physics.diffsim_rollout import (
    DifferentiableRollout,
    RIGID_DOF_SIZE,
    RIGID_POSE_SIZE,
    RolloutResult,
    _StepRecord,
    _real_dtype,
    step_with_substeps,
)

try:
    import torch
except ImportError as _torch_import_error:  # pragma: no cover
    raise ImportError(
        "superdex.physics.diffsim_torch requires PyTorch; install it with "
        "'pip install torch' (the CPU build is sufficient)"
    ) from _torch_import_error

diffsim = physics.diffsim

#: Field order of one row of ``contact_params``, matching the engine's
#: contact-parameter gradient layout.
CONTACT_PARAM_FIELDS = (
    "penalty_coefficient",
    "coulomb_friction_coefficient",
    "viscous_friction_coefficient",
    "normal_viscous_damping_coefficient",
)

#: Field order of one row of ``soft_materials``, matching the engine's soft
#: material gradient layout.
SOFT_MATERIAL_FIELDS = (
    "youngs_modulus",
    "poisson_ratio",
    "density",
    "mass_damping_coefficient",
)

__all__ = [
    "ArticulatedPoseObservation",
    "ContactForceObservation",
    "OrientationObservation",
    "SoftCentroidObservation",
    "CONTACT_PARAM_FIELDS",
    "PolicyRollout",
    "PolicyRolloutResult",
    "SOFT_MATERIAL_FIELDS",
    "TorchRollout",
    "TranslationObservation",
]


def _soft_material_get(params, field: str) -> float:
    if field in ("youngs_modulus", "poisson_ratio"):
        return float(getattr(params.neo_hookean, field))
    return float(getattr(params, field))


def _soft_material_set(params, field: str, value: float) -> None:
    """Write one field of a SoftMaterialParams; the elastic constants live in
    the sub-struct of the material's model (Lame-type models share the names)."""
    if field in ("youngs_modulus", "poisson_ratio"):
        model = {
            physics.SoftMaterialType.NEO_HOOKEAN: "neo_hookean",
            physics.SoftMaterialType.ST_VENANT_KIRCHHOFF: "st_venant_kirchhoff",
            physics.SoftMaterialType.LINEAR_ELASTIC: "linear_elastic",
        }.get(params.type)
        if model is None:
            raise ValueError(
                f"soft material type {params.type} has no Young's modulus / "
                "Poisson's ratio; soft_materials covers Lame-type materials only"
            )
        sub = getattr(params, model)
        setattr(sub, field, value)
        setattr(params, model, sub)
    else:
        setattr(params, field, value)


def _same_scene(actor, scene) -> bool:
    owner = actor.get_scene()
    return owner is scene or owner.get_handle() == scene.get_handle()


def _check_group(scene, actors, role: str) -> None:
    """Every actor of a declared input group belongs to ``scene`` and appears once.

    A bridge differentiates the scene it was built for: an actor of another scene (a
    matching name is not an identity - batched scenes reuse names) would be simulated in
    one scene and mutated or read in the other. Each entry of a group is an independent
    input block applied in order, so a repeated actor would have its earlier block
    overwritten while still receiving a gradient.
    """
    seen = set()
    for actor in actors:
        name = actor.get_name()
        if not _same_scene(actor, scene):
            raise ValueError(
                f"{role} actor {name!r} belongs to another scene "
                f"({actor.get_scene().get_name()!r}); a bridge differentiates the scene it "
                "was built for"
            )
        handle = actor.get_handle()
        if handle in seen:
            raise ValueError(
                f"{role} actor {name!r} is declared twice in the {role} group; each entry "
                "is an independent input block and a repeated actor's earlier block would be "
                "overwritten"
            )
        seen.add(handle)


def _require_double_precision() -> None:
    """The bridge's tensors are float64 because the engine simulates in double precision;
    the single-precision build would return float32-accurate gradients as float64 tensors."""
    if not physics.uses_double_precision():
        raise RuntimeError(
            "diffsim_torch requires the double-precision engine: install superdex-physics-fp64 "
            "and set SUPERDEX_PRECISION=double before importing superdex.physics (the "
            "single-precision build's gradients carry float32 accuracy and cannot be presented "
            "as float64 tensors)."
        )


def _as_numpy(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.dtype != torch.float64:
        raise TypeError(
            f"{name} must be float64 (the engine simulates in double "
            f"precision), got {value.dtype}"
        )
    if value.device.type != "cpu":
        raise TypeError(f"{name} must be a CPU tensor, got device {value.device}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    return value.detach().contiguous().numpy()


@dataclasses.dataclass
class _ForceEntry:
    actor: object
    dofs: np.ndarray  # int32 DoF indices receiving forces


class _RolloutLoss(torch.autograd.Function):
    """(controls, forces, gravity, contact, densities, initial_states,
    soft_materials) -> loss.

    Forward runs the rollout AND the adjoint sweep (the engine needs the
    captured step states); backward scales the stashed input gradients by
    ``grad_output``. The first argument is the owning bridge (non-tensor,
    non-differentiable).
    """

    @staticmethod
    def forward(
        ctx, bridge, controls, forces, gravity, contact, densities, initial_states, soft_materials
    ):
        grads = bridge._run(
            controls, forces, gravity, contact, densities, initial_states, soft_materials
        )
        ctx.saved_grads = grads
        return torch.tensor(bridge.last_result.loss, dtype=torch.float64)

    @staticmethod
    def backward(ctx, grad_output):
        scale = grad_output.detach().cpu().to(torch.float64)
        outs = tuple(
            None if g is None else scale * torch.from_numpy(g)
            for g in ctx.saved_grads
        )
        return (None, *outs)


class TorchRollout:
    """Torch-facing differentiable rollout with a fixed scene and loss."""

    def __init__(
        self,
        scene,
        dt: float,
        num_steps: int,
        *,
        control_actors: Sequence = (),
        force_actors: Sequence = (),
        contact_actors: Sequence = (),
        density_actors: Sequence = (),
        initial_state_actors: Sequence = (),
        soft_material_actors: Sequence = (),
        differentiate_gravity: bool = False,
        terminal_losses: Sequence = (),
        step_losses: Callable[[int], Sequence] | None = None,
        max_substep_levels: int = 0,
        substep_residual_tolerance: float | None = None,
    ):
        _require_double_precision()
        if not terminal_losses and step_losses is None:
            raise ValueError("provide terminal_losses and/or step_losses")
        self.scene = scene
        self.num_steps = num_steps
        self.differentiate_gravity = bool(differentiate_gravity)
        self._terminal_losses = list(terminal_losses)
        self._step_losses = step_losses
        # Failure-adaptive substepping is forwarded verbatim (see
        # DifferentiableRollout); ``last_result.split_steps`` reports what was split.
        self._rollout = DifferentiableRollout(
            scene,
            dt=dt,
            num_steps=num_steps,
            max_substep_levels=max_substep_levels,
            substep_residual_tolerance=substep_residual_tolerance,
        )
        # Actors are resolved by identity (handle), never by name: names label the
        # gradient dictionaries only. Every group is validated before any state is
        # captured or any parameter touched.
        for role, group in (
            ("control", control_actors),
            ("force", force_actors),
            ("contact", contact_actors),
            ("density", density_actors),
            ("initial-state", initial_state_actors),
            ("soft-material", soft_material_actors),
        ):
            _check_group(scene, group, role)
        by_handle = {entry.actor.get_handle(): entry for entry in self._rollout.entries}

        def _entry(actor, role: str):
            entry = by_handle.get(actor.get_handle())
            if entry is None:
                raise ValueError(
                    f"{role} actor {actor.get_name()!r} is not a dynamic actor of this "
                    "scene (static and nested-link actors are not supported)"
                )
            return entry

        self._control_actors = []
        for actor in control_actors:
            entry = _entry(actor, "control")
            if not entry.has_controller:
                raise ValueError(
                    f"control actor {entry.name!r} has no articulated pose "
                    "controller; controls are controller targets"
                )
            self._control_actors.append(entry)
        self.control_size = sum(e.dofs_size for e in self._control_actors)

        self._force_entries: list[_ForceEntry] = []
        for actor in force_actors:
            entry = _entry(actor, "force")
            if entry.soft:
                raise ValueError(
                    f"force actor {entry.name!r} is a soft actor; soft actors "
                    "take no external forces"
                )
            if entry.articulated:
                if not entry.force_dofs:
                    raise ValueError(
                        f"force actor {entry.name!r} has no single-DoF joints "
                        "to apply forces to"
                    )
                dofs = np.asarray(entry.force_dofs, dtype=np.int32)
            else:
                dofs = np.arange(RIGID_DOF_SIZE, dtype=np.int32)
            self._force_entries.append(_ForceEntry(entry.actor, dofs))
        self.force_size = sum(len(f.dofs) for f in self._force_entries)

        self._contact_actors = list(contact_actors)
        self._density_actors = list(density_actors)
        for actor in self._contact_actors:
            if actor.is_static():
                raise ValueError(
                    f"contact actor {actor.get_name()!r} is static; static "
                    "colliders are not part of back-propagated islands and "
                    "have no parameter gradients"
                )
        for actor in self._density_actors:
            if _entry(actor, "density").soft:
                raise ValueError(
                    f"density actor {actor.get_name()!r} is a soft actor; the "
                    "density gradient covers rigid-body inertia owners only "
                    "(soft material parameters are not differentiable yet)"
                )

        self._initial_state_entries = []
        for actor in initial_state_actors:
            entry = _entry(actor, "initial-state")
            if not entry.soft:
                raise NotImplementedError(
                    f"initial-state actor {entry.name!r} is not a soft actor; the "
                    "initial_states group covers soft actors only (rigid and "
                    "articulated initial-state gradients are available from "
                    "DifferentiableRollout results)"
                )
            self._initial_state_entries.append(entry)
        self.initial_state_size = 2 * sum(e.dofs_size for e in self._initial_state_entries)

        self._soft_material_actors = []
        for actor in soft_material_actors:
            entry = _entry(actor, "soft-material")
            if not entry.soft:
                raise ValueError(
                    f"soft-material actor {entry.name!r} is not a soft actor"
                )
            # Fail at construction, not at the first read, for unsupported models.
            _soft_material_set(
                actor.get_soft_material_params(), "youngs_modulus", 1.0
            )
            self._soft_material_actors.append(actor)

        # Names for gradient lookup (rollout results are keyed by name).
        self._control_names = [e.name for e in self._control_actors]

        self._state_init = scene.capture_state()
        self.last_result: RolloutResult | None = None

    def close(self) -> None:
        """Release the captured initial state (call before destroying the scene)."""
        if self._state_init is not None:
            self.scene.release_state(self._state_init)
            self._state_init = None

    # -- torch entry point -------------------------------------------------

    def __call__(
        self,
        *,
        controls: torch.Tensor | None = None,
        forces: torch.Tensor | None = None,
        gravity: torch.Tensor | None = None,
        contact_params: torch.Tensor | None = None,
        densities: torch.Tensor | None = None,
        initial_states: torch.Tensor | None = None,
        soft_materials: torch.Tensor | None = None,
    ) -> torch.Tensor:
        def check(name, value, declared, shape):
            if declared and value is None:
                raise ValueError(f"{name} is required (declared at construction)")
            if not declared and value is not None:
                raise ValueError(
                    f"{name} was not declared at construction; declare the "
                    "corresponding actors/flag to differentiate it"
                )

        check("controls", controls, bool(self._control_actors), None)
        check("forces", forces, bool(self._force_entries), None)
        check("gravity", gravity, self.differentiate_gravity, None)
        check("contact_params", contact_params, bool(self._contact_actors), None)
        check("densities", densities, bool(self._density_actors), None)
        check("initial_states", initial_states, bool(self._initial_state_entries), None)
        check("soft_materials", soft_materials, bool(self._soft_material_actors), None)
        return _RolloutLoss.apply(
            self,
            controls,
            forces,
            gravity,
            contact_params,
            densities,
            initial_states,
            soft_materials,
        )

    # -- engine side -------------------------------------------------------

    def _run(
        self, controls, forces, gravity, contact, densities, initial_states, soft_materials
    ):
        """Restore, apply inputs, rollout + adjoint sweep, read gradients.

        Returns the gradient arrays in the same order as the tensor inputs
        (``None`` for groups that are not differentiated).
        """
        if self._state_init is None:
            raise RuntimeError("TorchRollout is closed")
        controls_np = (
            _as_numpy("controls", controls, (self.num_steps, self.control_size))
            if self._control_actors
            else None
        )
        forces_np = (
            _as_numpy("forces", forces, (self.num_steps, self.force_size))
            if self._force_entries
            else None
        )
        gravity_np = (
            _as_numpy("gravity", gravity, (3,)) if self.differentiate_gravity else None
        )
        contact_np = (
            _as_numpy(
                "contact_params",
                contact,
                (len(self._contact_actors), len(CONTACT_PARAM_FIELDS)),
            )
            if self._contact_actors
            else None
        )
        densities_np = (
            _as_numpy("densities", densities, (len(self._density_actors),))
            if self._density_actors
            else None
        )
        initial_states_np = (
            _as_numpy("initial_states", initial_states, (self.initial_state_size,))
            if self._initial_state_entries
            else None
        )
        soft_materials_np = (
            _as_numpy(
                "soft_materials",
                soft_materials,
                (len(self._soft_material_actors), len(SOFT_MATERIAL_FIELDS)),
            )
            if self._soft_material_actors
            else None
        )

        self.scene.restore_state(self._state_init, False)
        if soft_materials_np is not None:
            for actor, row in zip(self._soft_material_actors, soft_materials_np):
                params = actor.get_soft_material_params()
                for field, value in zip(SOFT_MATERIAL_FIELDS, row):
                    _soft_material_set(params, field, float(value))
                actor.set_soft_material_params(params)
        if initial_states_np is not None:
            offset = 0
            for entry in self._initial_state_entries:
                n = entry.dofs_size
                entry.actor.set_displacements(
                    np.ascontiguousarray(initial_states_np[offset : offset + n])
                )
                entry.actor.set_node_velocities_local(
                    np.ascontiguousarray(initial_states_np[offset + n : offset + 2 * n])
                )
                offset += 2 * n
        if gravity_np is not None:
            self.scene.set_gravity(gravity_np.tolist())
        if contact_np is not None:
            for actor, row in zip(self._contact_actors, contact_np):
                params = actor.get_contact_params()
                for field, value in zip(CONTACT_PARAM_FIELDS, row):
                    setattr(params, field, float(value))
                actor.set_contact_params(params)
        if densities_np is not None:
            for actor, value in zip(self._density_actors, densities_np):
                actor.set_density(float(value))

        def apply_inputs(step: int) -> None:
            if controls_np is not None:
                offset = 0
                for entry in self._control_actors:
                    entry.actor.set_articulated_target_pose(
                        np.ascontiguousarray(
                            controls_np[step, offset : offset + entry.dofs_size]
                        )
                    )
                    offset += entry.dofs_size
            if forces_np is not None:
                offset = 0
                for force_entry in self._force_entries:
                    size = len(force_entry.dofs)
                    force_entry.actor.set_external_forces_on_dofs(
                        force_entry.dofs,
                        np.ascontiguousarray(forces_np[step, offset : offset + size]),
                    )
                    offset += size

        result = self._rollout.run(
            apply_inputs=apply_inputs
            if (controls_np is not None or forces_np is not None)
            else None,
            terminal_losses=self._terminal_losses,
            step_losses=self._step_losses,
        )
        self.last_result = result

        grad_controls = None
        if controls_np is not None:
            blocks = [
                result.gradients[name].control_targets.T
                for name in self._control_names
            ]
            grad_controls = np.ascontiguousarray(np.concatenate(blocks, axis=1))

        grad_forces = None
        if forces_np is not None:
            blocks = []
            for force_entry in self._force_entries:
                name = force_entry.actor.get_name()
                block = result.gradients[name].external_forces
                if block.shape[0] != len(force_entry.dofs):
                    raise RuntimeError(
                        f"force-gradient layout mismatch for actor {name!r}: "
                        f"driver produced {block.shape[0]} DoFs, bridge "
                        f"expected {len(force_entry.dofs)}"
                    )
                blocks.append(block.T)
            grad_forces = np.ascontiguousarray(np.concatenate(blocks, axis=1))

        grad_gravity = None
        if gravity_np is not None:
            grad_gravity = np.zeros(3)
            diffsim.set_gravity_backward(self.scene, grad_gravity)

        grad_contact = None
        if contact_np is not None:
            grad_contact = np.zeros((len(self._contact_actors), 4))
            for i, actor in enumerate(self._contact_actors):
                diffsim.set_contact_params_backward(actor, grad_contact[i])

        grad_densities = None
        if densities_np is not None:
            grad_densities = np.zeros(len(self._density_actors))
            row = np.zeros(1)
            for i, actor in enumerate(self._density_actors):
                diffsim.set_density_backward(actor, row)
                grad_densities[i] = row[0]

        grad_initial_states = None
        if initial_states_np is not None:
            blocks = []
            for entry in self._initial_state_entries:
                actor_grads = result.gradients[entry.name]
                blocks.append(actor_grads.initial_pose)
                blocks.append(actor_grads.initial_velocity)
            grad_initial_states = np.ascontiguousarray(np.concatenate(blocks))

        grad_soft_materials = None
        if soft_materials_np is not None:
            grad_soft_materials = np.zeros(
                (len(self._soft_material_actors), len(SOFT_MATERIAL_FIELDS))
            )
            for i, actor in enumerate(self._soft_material_actors):
                diffsim.set_soft_material_params_backward(actor, grad_soft_materials[i])

        return (
            grad_controls,
            grad_forces,
            grad_gravity,
            grad_contact,
            grad_densities,
            grad_initial_states,
            grad_soft_materials,
        )


# ---------------------------------------------------------------------------
# Closed-loop rollouts: a torch policy in the loop, gradients through the feedback
# ---------------------------------------------------------------------------


class ArticulatedPoseObservation:
    """The joint pose of an articulated actor (rotation-vector DoFs) as an observation."""

    def __init__(self, actor):
        if actor.get_type() != physics.ActorType.ARTICULATED:
            raise ValueError(f"{actor.get_name()!r} is not an articulated actor")
        self.actor = actor
        self.size = actor.get_num_dofs()

    def value(self) -> np.ndarray:
        pose = np.zeros(self.size)
        self.actor.get_articulated_pose(pose)
        return pose

    def accumulate_output_grad(self, grad: np.ndarray) -> None:
        diffsim.get_articulated_pose_backward(
            self.actor, np.ascontiguousarray(grad, dtype=_real_dtype())
        )


class OrientationObservation:
    """The world orientation of a rigid actor's center of mass as an observation: the unit
    quaternion in the engine's (x, y, z, w) order."""

    size = 4

    def __init__(self, actor):
        if actor.get_type() != physics.ActorType.RIGID:
            raise ValueError(f"{actor.get_name()!r} is not a rigid actor")
        self.actor = actor

    def value(self) -> np.ndarray:
        return np.asarray(self.actor.get_center_of_mass_transform().rotation.tolist(), dtype=np.float64)

    def accumulate_output_grad(self, grad: np.ndarray) -> None:
        full = np.zeros(RIGID_POSE_SIZE, dtype=_real_dtype())
        full[3:] = grad
        diffsim.get_center_of_mass_transform_backward(self.actor, full)


class ContactForceObservation:
    """The total world contact force on a rigid actor (standalone or an articulated link) as an
    observation: a tactile signal.

    The value is the engine's ``TOTAL_CONTACT_FORCE`` query, registered here (so construct the
    observation before the scene's first step): after step ``k`` it is the contact force of
    that step, and the loss gradient reaches the policy through
    :func:`superdex.physics.diffsim.get_contact_force_world_backward`, the engine's adjoint of
    the query (exact against static and moving colliders alike, see
    ``EngineContactForceAdjointTest``). The query has no value before any step, so the
    observation of the initial state is the force of a probe step taken from the initial state
    under the controls standing before the policy acts
    (:meth:`PolicyRollout.probe_initial_observations`, which restores the initial state
    afterwards): for a scene starting at rest that is the resting contact force, and in general
    the force a sensor would read during the first control interval. The initial observation
    is a constant of the rollout (it carries no gradient), like every other observation of the
    initial state.
    """

    size = 3
    requires_probe_step = True

    def __init__(self, actor):
        if actor.get_type() != physics.ActorType.RIGID:
            raise ValueError(f"{actor.get_name()!r} is not a rigid actor")
        if actor.is_static():
            raise ValueError(f"{actor.get_name()!r} is static: it takes no contact force")
        self.actor = actor
        self.query = actor.register_query(physics.QueryType.TOTAL_CONTACT_FORCE)

    def value(self) -> np.ndarray:
        return np.asarray(self.actor.get_contact_force_world(), dtype=np.float64)

    def accumulate_output_grad(self, grad: np.ndarray) -> None:
        diffsim.get_contact_force_world_backward(
            self.actor, np.ascontiguousarray(grad, dtype=_real_dtype())
        )


class SoftCentroidObservation:
    """The mean nodal displacement of a soft actor (the displacement of its node centroid, in
    the world frame) as an observation."""

    size = 3

    def __init__(self, actor):
        if actor.get_type() != physics.ActorType.SOFT:
            raise ValueError(f"{actor.get_name()!r} is not a soft actor")
        self.actor = actor
        self.num_nodes = actor.get_num_dofs() // 3

    def value(self) -> np.ndarray:
        displacements = np.asarray(self.actor.get_displacements(), dtype=np.float64)
        return displacements.reshape(self.num_nodes, 3).mean(axis=0)

    def accumulate_output_grad(self, grad: np.ndarray) -> None:
        nodal = np.tile(np.asarray(grad, dtype=_real_dtype()) / self.num_nodes, self.num_nodes)
        diffsim.get_displacements_backward(self.actor, np.ascontiguousarray(nodal))


class TranslationObservation:
    """The world translation of a rigid actor's center of mass as an observation."""

    size = 3

    def __init__(self, actor):
        if actor.get_type() != physics.ActorType.RIGID:
            raise ValueError(f"{actor.get_name()!r} is not a rigid actor")
        self.actor = actor

    def value(self) -> np.ndarray:
        return np.asarray(self.actor.get_center_of_mass_transform().translation, dtype=np.float64)

    def accumulate_output_grad(self, grad: np.ndarray) -> None:
        full = np.zeros(RIGID_POSE_SIZE, dtype=_real_dtype())
        full[:3] = grad
        diffsim.get_center_of_mass_transform_backward(self.actor, full)


@dataclasses.dataclass
class PolicyRolloutResult:
    loss: float
    fd_valid: bool
    max_adjoint_residual: float
    steps_swept: int
    split_steps: list


class _PolicyRolloutLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rollout, *params):
        loss, grads = rollout._run(params)
        ctx.grads = grads
        return torch.tensor(loss, dtype=torch.float64)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, *[grad_output * g for g in ctx.grads])


class PolicyRollout:
    """A closed-loop differentiable rollout: at every step a torch policy maps the current
    observation (and optionally the previous ``history - 1`` observations) to the
    pose-controller targets of the control actors, and the loss gradient with respect to
    the policy parameters is computed with the engine's adjoint, including the feedback
    path (the dependence of each control on the states the policy observed).

    ``policy`` is a callable (typically an ``nn.Module``) from a float64 tensor of shape
    ``(history * total_observation_size,)`` - the newest observation first - to a float64
    tensor of shape ``(total_control_size,)``: the concatenated pose-controller targets of
    ``control_actors`` followed by the external forces on the force DoFs of ``force_actors``
    (all six world-frame DoFs of a standalone rigid actor, the single-DoF joints of an
    articulated one; torque control of an articulated actor without a pose controller is the
    ``force_actors``-only case); every parameter of ``policy.parameters()`` that requires a
    gradient receives ``.grad`` from ``loss.backward()``. Observations are the ``value()`` /
    ``accumulate_output_grad(grad)`` objects above, taken from the pre-step state (the
    initial state for the first step, whose observation is treated as a constant: only
    the states produced by the rollout carry the feedback gradient). Losses follow the
    ``diffsim_rollout`` protocol. With ``time_feature`` the normalized step index
    ``step / num_steps`` is appended to the policy input (after the observation history),
    so a policy can carry a time-dependent baseline; it contributes no gradient. As in
    :class:`TorchRollout`, the forward and backward sweeps both run when the object is
    called, and every call restores the initial scene state captured at construction.

    The policy gradient is dL/dtheta = sum_k (du_k/dtheta)^T lambda_k, where lambda_k is
    the engine's gradient with respect to the targets applied at step k; the feedback
    enters through lambda_{x_{k-1}} += (do/dx_{k-1})^T (du_k/do)^T lambda_k, injected as
    an output gradient at the observed state before that step's back-propagation.
    """

    def __init__(
        self,
        scene,
        dt: float,
        num_steps: int,
        policy,
        observations: Sequence,
        control_actors: Sequence = (),
        force_actors: Sequence = (),
        *,
        history: int = 1,
        time_feature: bool = False,
        terminal_losses: Sequence = (),
        step_losses: Callable[[int], Sequence] | None = None,
        max_substep_levels: int = 0,
        substep_residual_tolerance: float | None = None,
    ):
        _require_double_precision()
        if not terminal_losses and step_losses is None:
            raise ValueError("provide terminal_losses and/or step_losses")
        if history < 1:
            raise ValueError("history must be at least 1")
        if not control_actors and not force_actors:
            raise ValueError("provide at least one control or force actor")
        if not observations:
            raise ValueError("provide at least one observation")
        self.scene = scene
        self.dt = dt
        self.num_steps = num_steps
        self.policy = policy
        self.observations = list(observations)
        self.history = history
        self.time_feature = bool(time_feature)
        self._terminal_losses = list(terminal_losses)
        self._step_losses = step_losses
        self._driver = DifferentiableRollout(
            scene,
            dt=dt,
            num_steps=num_steps,
            max_substep_levels=max_substep_levels,
            substep_residual_tolerance=substep_residual_tolerance,
        )
        # Actors are resolved by identity (handle), never by name (see _check_group); the
        # observations' actors must belong to this scene as well.
        _check_group(scene, control_actors, "control")
        _check_group(scene, force_actors, "force")
        for observation in self.observations:
            actor = getattr(observation, "actor", None)
            if actor is not None and not _same_scene(actor, scene):
                raise ValueError(
                    f"observation of actor {actor.get_name()!r} of another scene "
                    f"({actor.get_scene().get_name()!r}); a policy rollout observes the scene "
                    "it was built for"
                )
        by_handle = {entry.actor.get_handle(): entry for entry in self._driver.entries}
        self._control_entries = []
        for actor in control_actors:
            entry = by_handle.get(actor.get_handle())
            if entry is None or not entry.has_controller:
                raise ValueError(f"control actor {actor.get_name()!r} has no pose controller")
            self._control_entries.append(entry)
        self._force_entries = []
        for actor in force_actors:
            entry = by_handle.get(actor.get_handle())
            if entry is None or not entry.force_dofs:
                raise ValueError(f"force actor {actor.get_name()!r} takes no external forces")
            self._force_entries.append(entry)
        self.target_size = sum(e.dofs_size for e in self._control_entries)
        self.force_size = sum(len(e.force_dofs) for e in self._force_entries)
        self.control_size = self.target_size + self.force_size
        self.observation_size = sum(o.size for o in self.observations)
        self.input_size = history * self.observation_size + (1 if self.time_feature else 0)
        self._params = [p for p in policy.parameters() if p.requires_grad] if hasattr(policy, "parameters") else []
        self._state_init = scene.capture_state()
        self.last_result: PolicyRolloutResult | None = None

    def close(self) -> None:
        if self._state_init is not None:
            self.scene.release_state(self._state_init)
            self._state_init = None

    @property
    def initial_state(self):
        """The scene state captured at construction that every rollout starts from (a
        ``scene.restore_state`` argument, e.g. to replay the trained policy)."""
        if self._state_init is None:
            raise RuntimeError("PolicyRollout is closed")
        return self._state_init

    def __call__(self) -> torch.Tensor:
        if self._state_init is None:
            raise RuntimeError("PolicyRollout is closed")
        return _PolicyRolloutLoss.apply(self, *self._params)

    # -- pieces ------------------------------------------------------------

    def _observe(self) -> np.ndarray:
        return np.concatenate([np.asarray(o.value(), dtype=np.float64).reshape(-1) for o in self.observations])

    @property
    def needs_probe_step(self) -> bool:
        """Whether an observation reads a per-step engine query (a contact force), which has
        no value at the initial state until a probe step is taken."""
        return any(getattr(o, "requires_probe_step", False) for o in self.observations)

    def probe_initial_observations(self) -> None:
        """Fills the per-step queries at the initial state: one step from the initial state
        under the controls standing in the scene, then the initial state is restored (the
        queries keep the probe's values). Call it after ``scene.restore_state(initial_state)``
        and before reading the observations of the initial state, as :meth:`__call__` does;
        a no-op without such observations."""
        if self._state_init is None:
            raise RuntimeError("PolicyRollout is closed")
        if not self.needs_probe_step:
            return
        self.scene.step(self.dt)
        self.scene.restore_state(self._state_init, False)

    def _inject_observation_grad(self, grad: np.ndarray) -> None:
        offset = 0
        for o in self.observations:
            o.accumulate_output_grad(grad[offset : offset + o.size])
            offset += o.size

    def _apply_controls(self, controls: np.ndarray) -> None:
        offset = 0
        for entry in self._control_entries:
            entry.actor.set_articulated_target_pose(
                np.ascontiguousarray(controls[offset : offset + entry.dofs_size], dtype=np.float64)
            )
            offset += entry.dofs_size
        for entry in self._force_entries:
            size = len(entry.force_dofs)
            entry.actor.set_external_forces_on_dofs(
                np.asarray(entry.force_dofs, dtype=np.int32),
                np.ascontiguousarray(controls[offset : offset + size], dtype=np.float64),
            )
            offset += size

    def _read_target_grad(self, into: np.ndarray) -> None:
        """The targets' gradient of the step just back-propagated (read once per step: the
        engine folds the inherited substeps' gradients into the setting substep)."""
        real = _real_dtype()
        offset = 0
        for entry in self._control_entries:
            g = np.zeros(entry.dofs_size, dtype=real)
            diffsim.set_articulated_target_pose_backward(entry.actor, g)
            into[offset : offset + entry.dofs_size] += g
            offset += entry.dofs_size

    def _accumulate_force_grad(self, into: np.ndarray) -> None:
        """The external forces' gradient of the (sub)step just back-propagated (summed over
        the substeps of a step: the same forces act in each of them)."""
        real = _real_dtype()
        offset = self.target_size
        for entry in self._force_entries:
            size = len(entry.force_dofs)
            g = np.zeros(size, dtype=real)
            diffsim.set_external_forces_on_dofs_backward(
                entry.actor, np.asarray(entry.force_dofs, dtype=np.int32), g
            )
            into[offset : offset + size] += g
            offset += size

    def _run(self, params):
        scene = self.scene
        driver = self._driver
        scene.restore_state(self._state_init, False)
        self.probe_initial_observations()
        # Forward: policy in the loop; keep each step's torch graph for the reverse sweep.
        obs_history = [self._observe()] * self.history  # newest first
        graphs = []  # (obs_tensor, control_tensor) per step
        records: list[_StepRecord] = []
        # step_losses(step) is called once per step, on the step's final state; the sweep
        # differentiates those instances (see DifferentiableRollout.run).
        step_loss_lists: dict[int, list] = {}
        running_total = 0.0
        try:
            for step in range(self.num_steps):
                obs = np.concatenate(obs_history[: self.history])
                if self.time_feature:
                    obs = np.append(obs, step / self.num_steps)
                with torch.enable_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float64, requires_grad=True)
                    u_t = self.policy(obs_t)
                if u_t.shape != (self.control_size,) or u_t.dtype != torch.float64:
                    raise ValueError(
                        f"the policy must return a float64 tensor of shape ({self.control_size},), "
                        f"got {tuple(u_t.shape)} {u_t.dtype}"
                    )
                graphs.append((obs_t, u_t))
                self._apply_controls(u_t.detach().numpy())
                if driver.max_substep_levels == 0:
                    pre = scene.capture_state()
                    try:
                        scene.step(self.dt)
                        post = scene.capture_state()
                    except BaseException:
                        scene.release_state(pre)
                        raise
                    records.append(_StepRecord(step, self.dt, pre, post))
                else:
                    step_with_substeps(
                        scene,
                        self.dt,
                        driver.max_substep_levels,
                        driver.substep_residual_tolerance,
                        step=step,
                        on_substep=lambda pre, post, sub_dt, step=step: records.append(
                            _StepRecord(step, sub_dt, pre, post)
                        ),
                    )
                if self._step_losses is not None:
                    losses = list(self._step_losses(step))
                    step_loss_lists[step] = losses
                    running_total += sum(loss.value() for loss in losses)
                obs_history.insert(0, self._observe())
        except BaseException:
            driver._release(records)
            raise
        try:
            loss_value = sum(loss.value() for loss in self._terminal_losses) + running_total

            # Reverse sweep. pending[k] is the gradient to inject at the state after step k
            # (the observation the policy used for later steps); the initial state (k = -1) is
            # a constant.
            pending: dict[int, np.ndarray] = {}
            param_grads = [np.zeros(p.shape) for p in params]
            fd_valid = True
            max_residual = 0.0
            steps_swept = 0
            lambda_u = np.zeros(self.control_size)  # the current step's control gradient
            diffsim.reset_back_propagation(scene)
            for i in range(len(records) - 1, -1, -1):
                record = records[i]
                last_of_step = i == len(records) - 1 or records[i + 1].step != record.step
                first_of_step = i == 0 or records[i - 1].step != record.step
                diffsim.prepare_back_propagate(scene, record.post, record.pre)
                if i == len(records) - 1:
                    for loss in self._terminal_losses:
                        loss.accumulate_output_grad()
                if last_of_step:
                    for loss in step_loss_lists.get(record.step, ()):
                        loss.value()  # the restored step's derivative context
                        loss.accumulate_output_grad()
                    if record.step in pending:
                        self._inject_observation_grad(pending.pop(record.step))
                diffsim.back_propagate(scene)
                steps_swept += 1
                stats = diffsim.get_back_propagation_scene_stats(scene)
                fd_valid = fd_valid and stats.finite_diff_valid
                max_residual = max(max_residual, stats.residual_norm)
                if last_of_step:
                    lambda_u = np.zeros(self.control_size)
                self._accumulate_force_grad(lambda_u)
                if first_of_step:
                    self._read_target_grad(lambda_u)
                    obs_t, u_t = graphs[record.step]
                    grads = torch.autograd.grad(
                        u_t,
                        [obs_t, *params],
                        grad_outputs=torch.tensor(lambda_u, dtype=torch.float64),
                        allow_unused=True,
                    )
                    for accum, g in zip(param_grads, grads[1:]):
                        if g is not None:
                            accum += g.detach().numpy()
                    if grads[0] is not None:
                        obs_grad = grads[0].detach().numpy()
                        for h in range(self.history):
                            source = record.step - 1 - h  # the state after step `source`
                            if source < 0:
                                continue
                            block = obs_grad[h * self.observation_size : (h + 1) * self.observation_size]
                            pending[source] = pending.get(source, 0.0) + block
            split_steps = []
            for record in records:
                if split_steps and split_steps[-1][0] == record.step:
                    split_steps[-1] = (record.step, split_steps[-1][1] + 1)
                else:
                    split_steps.append((record.step, 1))
            self.last_result = PolicyRolloutResult(
                loss=loss_value,
                fd_valid=fd_valid,
                max_adjoint_residual=max_residual,
                steps_swept=steps_swept,
                split_steps=[entry for entry in split_steps if entry[1] > 1],
            )
            return loss_value, [torch.tensor(g, dtype=torch.float64) for g in param_grads]
        finally:
            # Every capture is released whatever raised: a loss, the policy, the adjoint.
            driver._release(records)
