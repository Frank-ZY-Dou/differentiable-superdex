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

"""Multi-step differentiable-rollout driver on top of ``superdex.physics.diffsim``.

The per-step adjoint API (``prepare_back_propagate`` / ``back_propagate`` and
the ``get_*_backward`` / ``set_*_backward`` functions) is deliberately
low-level. :class:`DifferentiableRollout` wraps it into one object that

- runs the forward rollout with per-step state capture (the checkpoints the
  reverse sweep replays from),
- accepts terminal and per-step (running) losses,
- runs the reverse sweep newest-step-first, optionally truncated to the last
  ``truncation_window`` steps (truncated BPTT),
- collects gradients w.r.t. the initial pose and velocity (for soft actors:
  the initial nodal displacements and velocities), per-step pose-controller
  targets, and per-step external forces (all six DoFs of a standalone rigid
  actor; the single-DoF joints of an articulated one; soft actors take no
  external forces),
- optionally clips each gradient block to a maximum L2 norm, and
- aggregates the solver diagnostics (finite-difference validity, worst
  adjoint residual, summed solve time) across the sweep.

Requirements are those of ``diffsim`` itself: rigid, articulated and
standalone soft actors (soft contact against static colliders only),
Backward Euler, double precision recommended (the driver also runs on the
single-precision build; gradients are then float32-accurate), and the
per-step protocol -
inputs are applied first, then the pre-step state is captured, then the scene
steps (no input changes in between).

Example::

    rollout = DifferentiableRollout(scene, dt=0.01, num_steps=32)
    result = rollout.run(
        apply_inputs=lambda step: actor.set_articulated_target_pose(plan[step]),
        terminal_losses=[my_loss],
    )
    grad_plan = result.control_gradients[actor_name]  # (num_dofs, num_steps)

Losses are objects with two methods: ``value() -> float`` and
``accumulate_output_grad() -> None`` (the latter calls diffsim output-backward
functions such as ``get_center_of_mass_transform_backward``); they are
evaluated on the live scene state.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence

import numpy as np

import superdex.physics as physics

diffsim = physics.diffsim

RIGID_POSE_SIZE = 7  # translation(3) + quaternion XYZW(4)
RIGID_DOF_SIZE = 6  # Lie tangent: d-translation(3) + d-rotation(3)


def _real_dtype():
    """numpy dtype of the engine's ``real``: the backward functions write into
    caller-provided buffers and reject a mismatching float width."""
    return np.float64 if physics.uses_double_precision() else np.float32

__all__ = [
    "ActorGradients",
    "DifferentiableRollout",
    "RolloutResult",
]


@dataclasses.dataclass
class _ActorEntry:
    actor: object
    name: str
    articulated: bool
    soft: bool
    dofs_size: int
    pose_size: int
    has_controller: bool
    force_dofs: list


def _collect_actors(scene) -> list[_ActorEntry]:
    actors = []
    scene.for_each_actor(actors.append)
    entries: list[_ActorEntry] = []
    for actor in actors:
        if actor.is_static() or actor.is_nested_link_actor():
            continue
        soft = actor.get_type() == physics.ActorType.SOFT
        articulated = actor.get_type() == physics.ActorType.ARTICULATED
        dofs = actor.get_num_dofs()
        has_controller = articulated and actor.has_articulated_pose_controller()
        if soft:
            # Soft actors carry no external forces and no controller; their
            # differentiable "inputs" are the initial nodal state read at the
            # end of the sweep (plus the scene-level parameter gradients).
            force_dofs = []
        elif articulated:
            force_dofs = []
            info = actor.get_articulated_shape_info()
            for entry in info.dof_info:
                if entry.get_size() == 1:
                    force_dofs.append(entry.offset)
        else:
            # Standalone rigid actors take world-frame external forces (DoFs
            # 0-2) and torques (DoFs 3-5); the backward reads the generalized
            # force adjoint for all six.
            force_dofs = list(range(RIGID_DOF_SIZE))
        name = actor.get_name()
        if any(e.name == name for e in entries):
            raise ValueError(
                f"duplicate dynamic actor name {name!r}: gradients are keyed by "
                "name, give each dynamic actor a unique one"
            )
        entries.append(
            _ActorEntry(
                actor,
                name,
                articulated,
                soft,
                dofs,
                dofs if (articulated or soft) else RIGID_POSE_SIZE,
                has_controller,
                force_dofs,
            )
        )
    return entries


def _clip(block: np.ndarray, max_norm: float | None) -> np.ndarray:
    if max_norm is None:
        return block
    norm = float(np.linalg.norm(block))
    if norm > max_norm and norm > 0.0:
        return block * (max_norm / norm)
    return block


@dataclasses.dataclass
class ActorGradients:
    """Gradients for one dynamic actor, in that actor's own coordinates.

    ``initial_pose`` uses the actor's external pose representation (7-vector
    translation + quaternion for rigid actors, joint pose for articulated
    ones, the local-frame nodal displacement vector for soft ones);
    ``initial_velocity`` stacks linear+angular for rigid actors, joint
    velocities for articulated ones, and the local-frame nodal velocity
    vector for soft ones. ``control_targets`` is ``(num_dofs, num_steps)``
    for actors with a pose controller, otherwise ``None``;
    ``external_forces`` is ``(len(force_dofs), num_steps)`` - ``force_dofs``
    being all six DoFs for a standalone rigid actor and the single-DoF joints
    for an articulated one - otherwise ``None`` (always ``None`` for soft
    actors, which take no external forces). Truncated sweeps leave the
    initial-state gradients as ``None`` (they would be incomplete) and only
    fill the steps the sweep visited.
    """

    initial_pose: np.ndarray | None
    initial_velocity: np.ndarray | None
    control_targets: np.ndarray | None
    external_forces: np.ndarray | None
    force_dofs: list


@dataclasses.dataclass
class RolloutResult:
    loss: float
    gradients: Mapping[str, ActorGradients]
    fd_valid: bool
    max_adjoint_residual: float
    solve_time_sec: float
    steps_swept: int

    @property
    def control_gradients(self) -> dict[str, np.ndarray]:
        return {
            name: g.control_targets
            for name, g in self.gradients.items()
            if g.control_targets is not None
        }


class DifferentiableRollout:
    """Forward rollout + reverse adjoint sweep with checkpointed states."""

    def __init__(
        self,
        scene,
        dt: float,
        num_steps: int,
        truncation_window: int | None = None,
        grad_clip_norm: float | None = None,
    ):
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if truncation_window is not None and truncation_window <= 0:
            raise ValueError("truncation_window must be positive when given")
        self.scene = scene
        self.dt = dt
        self.num_steps = num_steps
        self.truncation_window = truncation_window
        self.grad_clip_norm = grad_clip_norm
        self.entries = _collect_actors(scene)

    # -- pieces ------------------------------------------------------------

    def _forward(self, apply_inputs):
        pre, post = [], []
        for step in range(self.num_steps):
            if apply_inputs is not None:
                apply_inputs(step)
            pre.append(self.scene.capture_state())
            self.scene.step(self.dt)
            post.append(self.scene.capture_state())
        return pre, post

    def _read_step_input_grads(self, grads, step: int) -> None:
        real = _real_dtype()
        for entry, out in zip(self.entries, grads.values()):
            if entry.has_controller:
                g = np.zeros(entry.dofs_size, dtype=real)
                diffsim.set_articulated_target_pose_backward(entry.actor, g)
                out.control_targets[:, step] = g
            if entry.force_dofs:
                g = np.zeros(len(entry.force_dofs), dtype=real)
                diffsim.set_external_forces_on_dofs_backward(
                    entry.actor, np.asarray(entry.force_dofs, dtype=np.int32), g
                )
                out.external_forces[:, step] = g

    def _read_initial_grads(self, grads) -> None:
        real = _real_dtype()
        for entry, out in zip(self.entries, grads.values()):
            if entry.soft:
                gu = np.zeros(entry.dofs_size, dtype=real)
                diffsim.set_displacements_backward(entry.actor, gu)
                gv = np.zeros(entry.dofs_size, dtype=real)
                diffsim.set_node_velocities_local_backward(entry.actor, gv)
                out.initial_pose = gu.astype(np.float64)
                out.initial_velocity = gv.astype(np.float64)
            elif entry.articulated:
                gp = np.zeros(entry.dofs_size, dtype=real)
                diffsim.set_articulated_pose_from_joints_backward(entry.actor, gp)
                gv = np.zeros(entry.dofs_size, dtype=real)
                diffsim.set_articulated_joint_velocities_backward(entry.actor, gv)
                out.initial_pose = gp.astype(np.float64)
                out.initial_velocity = gv.astype(np.float64)
            else:
                gs = np.zeros(RIGID_POSE_SIZE, dtype=real)
                diffsim.set_center_of_mass_transform_backward(entry.actor, gs)
                gl = np.zeros(3, dtype=real)
                ga = np.zeros(3, dtype=real)
                diffsim.set_velocity_backward(entry.actor, gl, ga)
                out.initial_pose = gs.astype(np.float64)
                out.initial_velocity = np.concatenate([gl, ga]).astype(np.float64)

    # -- driver ------------------------------------------------------------

    def run(
        self,
        apply_inputs: Callable[[int], None] | None = None,
        terminal_losses: Sequence | None = None,
        step_losses: Callable[[int], Sequence] | None = None,
    ) -> RolloutResult:
        """One forward rollout followed by the reverse adjoint sweep.

        ``apply_inputs(step)`` is called before each forward step to set
        controller targets / external forces. ``terminal_losses`` are
        evaluated on the final state; ``step_losses(step)`` may return
        additional losses whose gradients are accumulated at that step of the
        reverse sweep (a running cost). At least one loss source is required.
        """
        if not terminal_losses and step_losses is None:
            raise ValueError("provide terminal_losses and/or step_losses")
        terminal_losses = list(terminal_losses or [])

        pre, post = self._forward(apply_inputs)

        loss_value = sum(loss.value() for loss in terminal_losses)

        grads = {
            entry.name: ActorGradients(
                initial_pose=None,
                initial_velocity=None,
                control_targets=(
                    np.zeros((entry.dofs_size, self.num_steps))
                    if entry.has_controller
                    else None
                ),
                external_forces=(
                    np.zeros((len(entry.force_dofs), self.num_steps))
                    if entry.force_dofs
                    else None
                ),
                force_dofs=list(entry.force_dofs),
            )
            for entry in self.entries
        }

        first_step = 1
        if self.truncation_window is not None:
            first_step = max(1, self.num_steps - self.truncation_window + 1)

        fd_valid = True
        max_residual = 0.0
        solve_time = 0.0
        steps_swept = 0

        diffsim.reset_back_propagation(self.scene)
        for i in range(self.num_steps, first_step - 1, -1):
            diffsim.prepare_back_propagate(self.scene, post[i - 1], pre[i - 1])
            if i == self.num_steps:
                for loss in terminal_losses:
                    loss.accumulate_output_grad()
            if step_losses is not None:
                for loss in step_losses(i - 1):
                    loss_value += loss.value()
                    loss.accumulate_output_grad()
            diffsim.back_propagate(self.scene)
            steps_swept += 1

            stats = diffsim.get_back_propagation_scene_stats(self.scene)
            fd_valid = fd_valid and stats.finite_diff_valid
            max_residual = max(max_residual, stats.residual_norm)
            solve_time += stats.solve_duration_sec

            self._read_step_input_grads(grads, i - 1)

        # Initial-state gradients are only complete when the sweep reached
        # the first step.
        if first_step == 1:
            self._read_initial_grads(grads)

        for handle in pre + post:
            self.scene.release_state(handle)

        if self.grad_clip_norm is not None:
            for out in grads.values():
                for field in (
                    "initial_pose",
                    "initial_velocity",
                    "control_targets",
                    "external_forces",
                ):
                    value = getattr(out, field)
                    if value is not None:
                        setattr(out, field, _clip(value, self.grad_clip_norm))

        return RolloutResult(
            loss=loss_value,
            gradients=grads,
            fd_valid=fd_valid,
            max_adjoint_residual=max_residual,
            solve_time_sec=solve_time,
            steps_swept=steps_swept,
        )
