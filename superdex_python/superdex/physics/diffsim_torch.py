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
- ``densities`` - per-actor mass densities, shape ``(num_density_actors,)``.

Design and contract:

- The forward and backward sweeps both run inside ``forward`` (the adjoint
  needs the step states while they are captured), so gradients are always
  computed; ``backward`` only scales the stashed gradients by the incoming
  ``grad_output``. Calling the bridge is therefore "loss + gradients", not
  "loss only".
- Tensors must be CPU ``float64`` with the exact documented shape - anything
  else raises. The engine simulates in double precision
  (``SUPERDEX_PRECISION=double``); mixed precision would silently degrade the
  gradients, so it is rejected rather than converted.
- A group declared at construction requires its tensor at every call, and a
  tensor for an undeclared group is rejected: there are no implicit defaults.
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

Known engine approximation (measured 2026-08-30, pinned by the test suite):
gradients w.r.t. the TORQUE components (DoFs 3-5) of a standalone rigid
actor's external forces carry a relative error of order the per-step rotation
angle - the engine's external-torque residual uses a merit function valid
near identity rotation steps (the error scales linearly with the applied
torque: 2.1e-4 at 0.15 N*m to 1.7e-3 at 1.2 N*m on a free cube), and contact
coupling can push the relative error on near-zero torque gradients to a few
percent (absolute error stayed below 1e-9 in all probes). Linear-force,
control, gravity, contact-parameter and density gradients pass
``torch.autograd.gradcheck`` exactly.

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
    RIGID_DOF_SIZE,
    DifferentiableRollout,
    RolloutResult,
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

__all__ = ["CONTACT_PARAM_FIELDS", "TorchRollout"]


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
    """(controls, forces, gravity, contact, densities) -> scalar loss.

    Forward runs the rollout AND the adjoint sweep (the engine needs the
    captured step states); backward scales the stashed input gradients by
    ``grad_output``. The first argument is the owning bridge (non-tensor,
    non-differentiable).
    """

    @staticmethod
    def forward(ctx, bridge, controls, forces, gravity, contact, densities):
        grads = bridge._run(controls, forces, gravity, contact, densities)
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
        differentiate_gravity: bool = False,
        terminal_losses: Sequence = (),
        step_losses: Callable[[int], Sequence] | None = None,
    ):
        if not terminal_losses and step_losses is None:
            raise ValueError("provide terminal_losses and/or step_losses")
        self.scene = scene
        self.num_steps = num_steps
        self.differentiate_gravity = bool(differentiate_gravity)
        self._terminal_losses = list(terminal_losses)
        self._step_losses = step_losses
        self._rollout = DifferentiableRollout(scene, dt=dt, num_steps=num_steps)
        by_name = {entry.name: entry for entry in self._rollout.entries}

        def _entry(actor, role: str):
            name = actor.get_name()
            if name not in by_name:
                raise ValueError(
                    f"{role} actor {name!r} is not a dynamic actor of this "
                    "scene (static and nested-link actors are not supported)"
                )
            return by_name[name]

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
        return _RolloutLoss.apply(
            self, controls, forces, gravity, contact_params, densities
        )

    # -- engine side -------------------------------------------------------

    def _run(self, controls, forces, gravity, contact, densities):
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

        self.scene.restore_state(self._state_init, False)
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

        return (grad_controls, grad_forces, grad_gravity, grad_contact, grad_densities)
