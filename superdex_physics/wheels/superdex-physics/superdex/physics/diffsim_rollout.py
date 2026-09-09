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
  actor; every joint dof of an articulated one; soft actors take no
  external forces),
- optionally clips each gradient block to a maximum L2 norm, and
- aggregates the solver diagnostics (finite-difference validity and the
  steps it flagged, worst adjoint residual, worst operator asymmetry, summed
  solve time) across the sweep, and
- optionally refines the time step where the forward Newton solve fails
  (``max_substep_levels`` > 0): the step is redone from its pre-step state as
  two half steps, recursively, each accepted substep becoming its own adjoint
  step with the parent step's inputs held fixed. The physical model is
  unchanged; only the time discretisation is locally finer, and the reverse
  sweep differentiates exactly the substeps that were solved (the split
  decision itself is treated as fixed). External-force gradients of the
  substeps are summed into the step's column; controller-target gradients are
  read once per step, at its first substep, because the engine propagates the
  gradient of an inherited (not re-set) target back to the step that set it.

Requirements are those of ``diffsim`` itself: rigid, articulated,
standalone soft and rod actors (a rod's initial state exposes its nodal
velocities, 4 per node), Backward Euler, double precision recommended (the
driver also runs on the single-precision build, with float32-accurate
gradients), and the per-step protocol: inputs are applied first, then the
pre-step state is captured, then the scene steps, with no input changes in
between.

Example::

    rollout = DifferentiableRollout(scene, dt=0.01, num_steps=32)
    result = rollout.run(
        apply_inputs=lambda step: actor.set_articulated_target_pose(plan[step]),
        terminal_losses=[my_loss],
    )
    grad_plan = result.gradients[actor.get_name()].control_targets  # (num_dofs, num_steps)

Losses are objects with two methods, ``value() -> float`` and
``accumulate_output_grad() -> None``; the latter calls the diffsim output
adjoints such as ``get_center_of_mass_transform_backward``. See
:meth:`DifferentiableRollout.run` for when each is called.
"""

from __future__ import annotations

import dataclasses
import math
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
    rod: bool
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
        rod = actor.get_type() == physics.ActorType.ROD
        articulated = actor.get_type() == physics.ActorType.ARTICULATED
        dofs = actor.get_num_dofs()
        has_controller = articulated and actor.has_articulated_pose_controller()
        if soft or rod:
            # Soft and rod actors carry no differentiable external forces and no
            # controller; their differentiable "inputs" are the initial nodal state
            # read at the end of the sweep (plus the scene-level parameter gradients).
            force_dofs = []
        elif articulated:
            # Every joint dof takes an external force: the generalized force of a revolute or
            # prismatic joint, and the force and torque (in the joint's outer frame) of a Free
            # or Spherical joint; the backward reads the generalized-force adjoint for all.
            force_dofs = list(range(dofs))
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
                rod,
                dofs,
                dofs if (articulated or soft or rod) else RIGID_POSE_SIZE,
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


class ForwardSolveError(RuntimeError):
    """The forward Newton solve of one step did not converge.

    Raised by :func:`step_with_substeps` (and hence by rollouts with
    ``max_substep_levels`` > 0) when even the finest substep level fails, and by
    every rollout, substepping or not, when a step's Newton residual is not a
    finite number.
    """

    def __init__(self, step: int, dt: float, status, residual_norm: float, iterations: int):
        self.step = step
        self.dt = dt
        self.status = status
        self.residual_norm = residual_norm
        self.iterations = iterations
        super().__init__(
            f"forward Newton solve did not converge at step {step} with dt={dt:g} "
            f"(status {status.name}, residual {residual_norm:.2e}, "
            f"{iterations} iterations)"
        )


def _solver_stats(scene):
    return scene.get_solver_stats()


def forward_solve_failed(scene, residual_tolerance: float) -> bool:
    """Whether the last ``scene.step`` ended without convergence and with a
    residual above ``residual_tolerance``, or with a residual that is not a
    finite number.

    A solve that stopped on round-off with a residual below the tolerance
    counts as converged; a solve that hit the iteration limit or diverged with a
    larger residual is a failure. A NaN or infinite residual is a failure
    whatever the status says: a comparison with NaN is false, so the threshold
    alone would accept it (before 2026-09-08 it did).
    """
    stats = _solver_stats(scene)
    residual = float(stats.residual_norm)
    if not math.isfinite(residual):
        return True
    return (
        stats.convergence_status != physics.ConvergenceStatus.CONVERGED
        and residual > residual_tolerance
    )


def _raise_if_not_finite(scene, step: int, dt: float) -> None:
    """A step whose Newton residual is not a finite number left the scene in a
    corrupt state: an error even for a rollout that does not inspect
    convergence otherwise."""
    stats = _solver_stats(scene)
    if not math.isfinite(float(stats.residual_norm)):
        raise ForwardSolveError(
            step, dt, stats.convergence_status, stats.residual_norm, stats.max_non_linear_iters
        )


def _step_adaptive(
    scene, dt: float, level: int, max_levels: int, residual_tolerance: float,
    step: int, on_substep, pre,
) -> list[float]:
    """One (sub)step from the captured pre-state ``pre``, which this call owns until it is
    handed to ``on_substep`` together with the post-state (or released): whatever raises
    before that - the native step, the convergence check, the restore before a subdivision,
    the post-state capture, the callback - releases what has not been handed over. A
    subdivision hands ``pre`` to its first half and captures a pre-state of its own for the
    second, so no handle has two owners."""
    try:
        scene.step(dt)
        failed = forward_solve_failed(scene, residual_tolerance)
        if failed:
            stats = _solver_stats(scene)
            if level >= max_levels:
                raise ForwardSolveError(
                    step,
                    dt,
                    stats.convergence_status,
                    stats.residual_norm,
                    stats.max_non_linear_iters,
                )
            scene.restore_state(pre, False)
    except BaseException:
        scene.release_state(pre)
        raise
    if failed:
        half = dt / 2.0
        taken = _step_adaptive(
            scene, half, level + 1, max_levels, residual_tolerance, step, on_substep, pre
        )
        taken += _step_adaptive(
            scene,
            half,
            level + 1,
            max_levels,
            residual_tolerance,
            step,
            on_substep,
            scene.capture_state(),
        )
        return taken
    try:
        post = scene.capture_state()
    except BaseException:
        scene.release_state(pre)
        raise
    if on_substep is None:
        scene.release_state(pre)
        scene.release_state(post)
        return [dt]
    try:
        on_substep(pre, post, dt)
    except BaseException:
        scene.release_state(pre)
        scene.release_state(post)
        raise
    return [dt]


def step_with_substeps(
    scene,
    dt: float,
    max_levels: int,
    residual_tolerance: float,
    step: int = 0,
    on_substep: Callable | None = None,
) -> list[float]:
    """Advance ``scene`` by ``dt``; if the forward Newton solve fails, redo the
    step from its pre-step state as two half steps (recursively, at most
    ``max_levels`` times).

    Returns the substep sizes actually taken (``[dt]`` when the plain step
    converged). ``on_substep(pre, post, sub_dt)`` receives the captured states
    around every accepted substep and owns them afterwards; without it the
    captures are released. Inputs (controller targets, external forces) are
    part of the captured state and therefore stay fixed across the substeps.
    Raises :class:`ForwardSolveError` when the finest level fails too; ``step``
    only labels that error.
    """
    if max_levels < 0:
        raise ValueError("max_levels must be non-negative")
    if not residual_tolerance > 0.0:
        raise ValueError("residual_tolerance must be positive")
    return _step_adaptive(
        scene, dt, 0, max_levels, residual_tolerance, step, on_substep,
        scene.capture_state(),
    )


@dataclasses.dataclass
class _StepRecord:
    step: int  # 0-based rollout step this (sub)step belongs to
    dt: float
    pre: object
    post: object


@dataclasses.dataclass
class ActorGradients:
    """Gradients for one dynamic actor, in that actor's own coordinates.

    ``initial_pose`` uses the actor's external pose representation (7-vector
    translation + quaternion for rigid actors, joint pose for articulated
    ones, the local-frame nodal displacement vector for soft ones);
    ``initial_velocity`` stacks linear+angular for rigid actors, joint
    velocities for articulated ones, and the local-frame nodal velocity
    vector for soft ones. For an articulated actor ``initial_pose`` holds the
    joint velocities fixed: in a differentiable scene the pose setters derive
    the link velocities from the joint velocities at the pose, as the velocity
    setter does, so the gradient does not depend on the order of the two.
    ``control_targets`` is ``(num_dofs, num_steps)``
    for actors with a pose controller, otherwise ``None``;
    ``external_forces`` is ``(len(force_dofs), num_steps)`` - ``force_dofs``
    being all six DoFs for a standalone rigid actor and every joint dof for an
    articulated one - otherwise ``None`` (always ``None`` for soft
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
    # Diagnostics of the adjoint solves across the sweep (both only meaningful with
    # ``BackPropagationSolverParams.validate_finite_diff`` set): the largest relative
    # asymmetry of the adjoint operator measured by the engine's symmetry probe, and
    # the 0-based steps whose finite-difference self-check failed.
    max_hessian_asymmetry: float = 0.0
    flagged_steps: list = dataclasses.field(default_factory=list)
    # Largest outer-iteration count of one adjoint solve, and the number of island
    # solves (over the sweep) whose PCG aborted and fell back to MINRES.
    max_outer_iters: int = 0
    minres_fallbacks: int = 0
    # Failure-adaptive substepping (``max_substep_levels`` > 0): the 0-based steps that
    # were split, with the number of substeps each was solved in, and the total number
    # of solver steps of the forward rollout (``num_steps`` when nothing was split).
    split_steps: list = dataclasses.field(default_factory=list)
    num_solver_steps: int = 0

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
        max_substep_levels: int = 0,
        substep_residual_tolerance: float | None = None,
    ):
        """``max_substep_levels`` > 0 enables failure-adaptive substepping: a
        step whose Newton solve ends without convergence and with a residual
        above ``substep_residual_tolerance`` (required then; typically a few
        times the solver tolerance) is redone as two half steps, recursively up
        to ``max_substep_levels`` halvings, and :class:`ForwardSolveError` is
        raised if the finest level fails too. With the default 0 the forward
        rollout never inspects convergence.

        ``truncation_window`` limits the reverse sweep to the last that many
        steps (truncated backpropagation through time): the input gradients of
        the earlier steps stay zero and the initial-state gradients are
        withheld, while the reported loss still sums the running costs of every
        step - the window changes the gradient, never the objective.
        """
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if truncation_window is not None and truncation_window <= 0:
            raise ValueError("truncation_window must be positive when given")
        if max_substep_levels < 0:
            raise ValueError("max_substep_levels must be non-negative")
        if max_substep_levels > 0:
            if substep_residual_tolerance is None or not substep_residual_tolerance > 0.0:
                raise ValueError(
                    "substep_residual_tolerance must be a positive float when "
                    "max_substep_levels > 0"
                )
        elif substep_residual_tolerance is not None:
            raise ValueError(
                "substep_residual_tolerance has no effect without max_substep_levels > 0"
            )
        self.scene = scene
        self.dt = dt
        self.num_steps = num_steps
        self.truncation_window = truncation_window
        self.grad_clip_norm = grad_clip_norm
        self.max_substep_levels = max_substep_levels
        self.substep_residual_tolerance = substep_residual_tolerance
        self.entries = _collect_actors(scene)

    # -- pieces ------------------------------------------------------------

    def _forward(self, apply_inputs, on_step_end=None) -> list[_StepRecord]:
        """The forward rollout; ``on_step_end(step)`` runs on each step's final state."""
        records: list[_StepRecord] = []
        try:
            for step in range(self.num_steps):
                if apply_inputs is not None:
                    apply_inputs(step)
                if self.max_substep_levels == 0:
                    pre = self.scene.capture_state()
                    try:
                        self.scene.step(self.dt)
                        _raise_if_not_finite(self.scene, step, self.dt)
                        post = self.scene.capture_state()
                    except BaseException:
                        self.scene.release_state(pre)
                        raise
                    records.append(_StepRecord(step, self.dt, pre, post))
                else:
                    step_with_substeps(
                        self.scene,
                        self.dt,
                        self.max_substep_levels,
                        self.substep_residual_tolerance,
                        step=step,
                        on_substep=lambda pre, post, sub_dt, step=step: records.append(
                            _StepRecord(step, sub_dt, pre, post)
                        ),
                    )
                if on_step_end is not None:
                    on_step_end(step)
        except BaseException:
            self._release(records)
            raise
        return records

    def _release(self, records: Sequence[_StepRecord]) -> None:
        for record in records:
            self.scene.release_state(record.pre)
            self.scene.release_state(record.post)

    def _read_step_input_grads(self, grads, step: int, read_targets: bool = True) -> None:
        real = _real_dtype()
        for entry, out in zip(self.entries, grads.values()):
            if entry.has_controller and read_targets:
                g = np.zeros(entry.dofs_size, dtype=real)
                diffsim.set_articulated_target_pose_backward(entry.actor, g)
                out.control_targets[:, step] += g
            if entry.force_dofs:
                g = np.zeros(len(entry.force_dofs), dtype=real)
                diffsim.set_external_forces_on_dofs_backward(
                    entry.actor, np.asarray(entry.force_dofs, dtype=np.int32), g
                )
                out.external_forces[:, step] += g

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
            elif entry.rod:
                # Rods take initial velocities (4 per node, incl. the twist rate) but no
                # initial displacements through the public API, so only the velocity
                # gradient is read back.
                gv = np.zeros(entry.dofs_size, dtype=real)
                diffsim.set_node_velocities_local_backward(entry.actor, gv)
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

        Loss protocol: ``step_losses(step)`` is called exactly once per step,
        on that step's final state during the forward rollout, and the
        instances it returns are the ones the sweep differentiates - a factory
        may sample or consume data (targets, weights, minibatches) and still
        defines a single objective. ``value()`` is called on the state the
        loss refers to (each step's final state for a running cost, the final
        state for a terminal loss) and the values sum to the objective; during
        the reverse sweep, with the step's state restored, ``value()`` is
        called again right before ``accumulate_output_grad()``, so a loss may
        cache whatever its gradient needs in ``value()`` - the cache always
        belongs to the state being differentiated. The factory may return
        fresh instances at every call or shared ones.
        """
        if not terminal_losses and step_losses is None:
            raise ValueError("provide terminal_losses and/or step_losses")
        terminal_losses = list(terminal_losses or [])

        # ``step_losses(step)`` is called exactly once per step, on the step's final state
        # during the forward rollout: the instances it returns are evaluated there (the
        # objective is the whole trajectory's whatever the truncation window) and kept for
        # the sweep, which differentiates those same instances - a factory that samples or
        # consumes data (targets, weights, minibatches) defines one objective, and its
        # gradient is that objective's.
        step_loss_lists: dict[int, list] = {}
        running_total = [0.0]

        def on_step_end(step: int) -> None:
            if step_losses is not None:
                losses = list(step_losses(step))
                step_loss_lists[step] = losses
                running_total[0] += sum(loss.value() for loss in losses)

        records = self._forward(apply_inputs, on_step_end)
        try:
            return self._sweep(records, terminal_losses, step_loss_lists, running_total[0])
        finally:
            # Every capture is released whatever raised: a loss, the adjoint, a reader.
            self._release(records)

    def _sweep(
        self,
        records: list[_StepRecord],
        terminal_losses: list,
        step_loss_lists: dict[int, list],
        running_total: float,
    ) -> RolloutResult:
        """The reverse adjoint sweep over ``records`` (owned by the caller)."""
        loss_value = sum(loss.value() for loss in terminal_losses) + running_total

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
        max_asymmetry = 0.0
        flagged_steps: list[int] = []
        max_outer_iters = 0
        minres_fallbacks = 0
        solve_time = 0.0
        steps_swept = 0

        # Reverse sweep over the solver steps, newest first. A step's losses are
        # evaluated on its final state, i.e. at the last of its substeps.
        diffsim.reset_back_propagation(self.scene)
        for k in range(len(records) - 1, -1, -1):
            record = records[k]
            last_of_step = k == len(records) - 1 or records[k + 1].step != record.step
            first_of_step = k == 0 or records[k - 1].step != record.step
            if record.step < first_step - 1:
                # Before the truncation window: no adjoint (the running costs of these steps
                # were summed during the forward rollout).
                break
            diffsim.prepare_back_propagate(self.scene, record.post, record.pre)
            if k == len(records) - 1:
                for loss in terminal_losses:
                    loss.accumulate_output_grad()
            if last_of_step:
                for loss in step_loss_lists.get(record.step, ()):
                    # The forward rollout's instances. value() right before the gradient,
                    # on the restored step: a loss may cache its derivative context in
                    # value() (the objective took the live values; this one is discarded).
                    loss.value()
                    loss.accumulate_output_grad()
            diffsim.back_propagate(self.scene)
            steps_swept += 1

            stats = diffsim.get_back_propagation_scene_stats(self.scene)
            fd_valid = fd_valid and stats.finite_diff_valid
            if not stats.finite_diff_valid and record.step not in flagged_steps:
                flagged_steps.append(record.step)
            max_residual = max(max_residual, stats.residual_norm)
            max_asymmetry = max(max_asymmetry, stats.hessian_asymmetry)
            max_outer_iters = max(max_outer_iters, stats.max_outer_iters)
            minres_fallbacks += stats.num_minres_fallbacks
            solve_time += stats.solve_duration_sec

            # Targets are set once per step and inherited by its later substeps; the
            # engine folds the inherited substeps' target gradients into the first
            # substep's, so they are read there only. Forces enter every substep.
            self._read_step_input_grads(grads, record.step, read_targets=first_of_step)

        # Initial-state gradients are only complete when the sweep reached
        # the first step.
        if first_step == 1:
            self._read_initial_grads(grads)

        split_steps = []
        for record in records:
            if split_steps and split_steps[-1][0] == record.step:
                split_steps[-1] = (record.step, split_steps[-1][1] + 1)
            else:
                split_steps.append((record.step, 1))
        split_steps = [entry for entry in split_steps if entry[1] > 1]

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
            max_hessian_asymmetry=max_asymmetry,
            flagged_steps=flagged_steps[::-1],
            max_outer_iters=max_outer_iters,
            minres_fallbacks=minres_fallbacks,
            split_steps=split_steps,
            num_solver_steps=len(records),
        )
