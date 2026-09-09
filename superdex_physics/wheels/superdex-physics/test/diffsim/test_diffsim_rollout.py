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

"""Tests for :mod:`superdex.physics.diffsim_rollout`.

- exact equivalence with the low-level harness sweep on a controller scene
  (same ops in the same order must give bitwise-close gradients);
- running (per-step) losses validated against central finite differences of
  the summed rollout objective;
- truncation semantics: the newest step's control gradient is unaffected,
  initial-state gradients are withheld, steps before the window stay zero;
- gradient clipping caps every block's L2 norm;
- soft actors: the driver's initial nodal displacement/velocity gradients on
  a soft cube in contact with a static plane equal the hand-written per-step
  sweep bit-for-bit, a driver running loss on a soft cube matches central
  finite differences, and truncation withholds the initial-state gradients;
- failure-adaptive substepping: with the failure predicate injected, the driver
  splits exactly the declared steps (two halves, then quarters), sums the
  substeps' gradients into the parent step's control / force columns, and its
  gradients match central finite differences of a manual rollout with the same
  substep schedule; the finest level failing raises ForwardSolveError; the
  arguments are validated;
- single precision: RunningLossTest also runs on the float32 build (5% tolerance);
  every other class requires SUPERDEX_PRECISION=double;
- variable step sizes: the low-level per-step adjoint chained across steps of
  different dt (as substepping produces) matches finite differences for a free
  rigid body's initial velocity, for joint forces and for pose-controller
  targets on a pendulum (the engine rescales the previous-delta adjoint by
  dt_k / dt_{k-1}, runs the adjoint's pre-step with the current step's dt, and
  re-expresses a finite-difference angular velocity for a changed step size so
  that the rotation predictor stays a rotation); a spinning free body keeps its
  angular rate across a step-size change.

Requires SUPERDEX_PRECISION=double except where noted.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np
import superdex.physics as physics
from superdex.physics import diffsim_rollout
from superdex.physics.diffsim_rollout import DifferentiableRollout, ForwardSolveError

from . import scenes
from .harness import (
    ArticulatedPoseErrorLoss,
    DisplacementErrorLoss,
    GradientCheckCase,
    TranslationErrorLoss,
    configure_for_differentiability,
)

diffsim = physics.diffsim

_NUM_WORKER_THREADS = int(os.environ.get("SUPERDEX_DIFFSIM_TEST_THREADS", "0"))
DT = 0.01
NUM_STEPS = 6


DOUBLE = physics.uses_double_precision()
double_only = unittest.skipUnless(DOUBLE, "requires SUPERDEX_PRECISION=double")


def setUpModule() -> None:
    physics.initialize(num_worker_threads=_NUM_WORKER_THREADS)


def tearDownModule() -> None:
    if physics.is_initialized():
        physics.shutdown()


def _controller_setup():
    """Pendulum + controller with a deterministic input schedule."""
    scene, chain = scenes.pendulum(with_controller=True)
    configure_for_differentiability(scene)
    num_dofs = chain.get_num_dofs()
    pose0 = np.zeros(num_dofs)
    chain.get_articulated_pose(pose0)
    targets = np.stack(
        [pose0 + 0.5 * DT * j for j in range(NUM_STEPS)], axis=1
    )  # (dofs, steps)
    forces = np.stack(
        [DT * j * np.ones(num_dofs) for j in range(NUM_STEPS)], axis=1
    )
    force_dofs = np.arange(num_dofs, dtype=np.int32)

    def apply_inputs(step: int) -> None:
        if step == 0:
            chain.set_articulated_target_velocity(np.zeros(num_dofs))
        chain.set_articulated_target_pose(np.ascontiguousarray(targets[:, step]))
        chain.set_external_forces_on_dofs(
            force_dofs, np.ascontiguousarray(forces[:, step])
        )

    return scene, chain, targets, forces, apply_inputs


@double_only
class RolloutEquivalenceTest(unittest.TestCase):
    """Driver gradients must match the low-level harness sweep exactly."""

    def test_matches_harness_sweep(self) -> None:
        ref_loss = np.array([0.4, -0.2])

        # Reference: the harness's own sweep on an identically-built scene.
        scene_a, chain_a = scenes.pendulum(with_controller=True)
        self.addCleanup(physics.destroy_scene, scene_a)
        case = GradientCheckCase(
            scene_a,
            [ArticulatedPoseErrorLoss(chain_a, ref_loss)],
            num_steps=NUM_STEPS,
            dt=DT,
            control_speed=0.5,
            force_speed=1.0,
        )
        ref = case.run_backward()

        # Driver on a fresh identical scene with the same input schedule.
        scene_b, chain_b, _targets, _forces, apply_inputs = _controller_setup()
        self.addCleanup(physics.destroy_scene, scene_b)
        rollout = DifferentiableRollout(scene_b, dt=DT, num_steps=NUM_STEPS)
        result = rollout.run(
            apply_inputs=apply_inputs,
            terminal_losses=[ArticulatedPoseErrorLoss(chain_b, ref_loss)],
        )

        self.assertTrue(result.fd_valid)
        self.assertEqual(result.steps_swept, NUM_STEPS)
        grads = result.gradients["chain"]
        np.testing.assert_allclose(
            grads.control_targets, ref["control"], rtol=1e-9, atol=1e-14
        )
        np.testing.assert_allclose(
            grads.external_forces, ref["force"], rtol=1e-9, atol=1e-14
        )
        np.testing.assert_allclose(
            grads.initial_pose, ref["init_pose"], rtol=1e-9, atol=1e-14
        )
        np.testing.assert_allclose(
            grads.initial_velocity, ref["init_vel"], rtol=1e-9, atol=1e-14
        )


class RunningLossTest(unittest.TestCase):
    """Per-step losses validated against finite differences of the summed cost.

    Also the single-precision smoke test of the driver: on the float32 build the
    gradient buffers are float32 and the check uses a coarser difference quotient
    (eps 1e-3, 5% tolerance) - float32-accurate gradients, as documented.
    """

    FD_EPS = 1e-6 if DOUBLE else 1e-3
    TOL = 1e-4 if DOUBLE else 5e-2

    def test_running_translation_loss_vs_fd(self) -> None:
        scene, cube = scenes.rigid_free()
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        loss = TranslationErrorLoss(cube)
        state_init = scene.capture_state()

        rollout = DifferentiableRollout(scene, dt=DT, num_steps=NUM_STEPS)
        result = rollout.run(step_losses=lambda step: [loss])
        grads = result.gradients["cube"]

        def objective() -> float:
            total = 0.0
            for _ in range(NUM_STEPS):
                scene.step(DT)
                total += loss.value()
            return total

        # FD w.r.t. initial velocity (linear + angular).
        fd = np.zeros(6)
        for j in range(6):
            values = []
            for sign in (+1.0, -1.0):
                scene.restore_state(state_init, False)
                lin = np.asarray(cube.get_linear_velocity(), dtype=np.float64)
                ang = np.asarray(cube.get_angular_velocity(), dtype=np.float64)
                delta = np.zeros(3)
                delta[j % 3] = sign * self.FD_EPS
                if j < 3:
                    lin = lin + delta
                else:
                    ang = ang + delta
                cube.set_velocity(lin, ang)
                values.append(objective())
            fd[j] = (values[0] - values[1]) / (2.0 * self.FD_EPS)
        scene.release_all_states()

        rel = np.linalg.norm(grads.initial_velocity - fd) / np.linalg.norm(fd)
        self.assertLessEqual(
            rel,
            self.TOL,
            f"running-loss gradient mismatch: analytic={grads.initial_velocity}, fd={fd}",
        )


@double_only
class TruncationAndClippingTest(unittest.TestCase):
    def test_truncation_semantics(self) -> None:
        scene_full, _, _, _, apply_full = _controller_setup()
        self.addCleanup(physics.destroy_scene, scene_full)
        ref_loss = np.array([0.4, -0.2])
        full = DifferentiableRollout(scene_full, dt=DT, num_steps=NUM_STEPS).run(
            apply_inputs=apply_full,
            terminal_losses=[
                ArticulatedPoseErrorLoss(
                    scene_full_actor(scene_full), ref_loss
                )
            ],
        )

        scene_trunc, _, _, _, apply_trunc = _controller_setup()
        self.addCleanup(physics.destroy_scene, scene_trunc)
        window = 3
        trunc = DifferentiableRollout(
            scene_trunc, dt=DT, num_steps=NUM_STEPS, truncation_window=window
        ).run(
            apply_inputs=apply_trunc,
            terminal_losses=[
                ArticulatedPoseErrorLoss(
                    scene_full_actor(scene_trunc), ref_loss
                )
            ],
        )

        self.assertEqual(trunc.steps_swept, window)
        g_full = full.gradients["chain"]
        g_trunc = trunc.gradients["chain"]
        # The newest step is unaffected by truncation.
        np.testing.assert_allclose(
            g_trunc.control_targets[:, -1],
            g_full.control_targets[:, -1],
            rtol=1e-9,
            atol=1e-14,
        )
        # Steps before the window stay zero; initial-state grads are withheld.
        self.assertTrue(
            np.all(g_trunc.control_targets[:, : NUM_STEPS - window] == 0.0)
        )
        self.assertIsNone(g_trunc.initial_pose)
        self.assertIsNone(g_trunc.initial_velocity)

    def test_grad_clipping(self) -> None:
        scene, _, _, _, apply_inputs = _controller_setup()
        self.addCleanup(physics.destroy_scene, scene)
        max_norm = 1e-6
        result = DifferentiableRollout(
            scene, dt=DT, num_steps=NUM_STEPS, grad_clip_norm=max_norm
        ).run(
            apply_inputs=apply_inputs,
            terminal_losses=[
                ArticulatedPoseErrorLoss(
                    scene_full_actor(scene), np.array([0.4, -0.2])
                )
            ],
        )
        g = result.gradients["chain"]
        for block in (
            g.initial_pose,
            g.initial_velocity,
            g.control_targets,
            g.external_forces,
        ):
            self.assertLessEqual(float(np.linalg.norm(block)), max_norm * (1 + 1e-12))


@double_only
class RigidExternalForceGradientTest(unittest.TestCase):
    """The driver's rigid-actor force gradients (all six DoFs) vs central FD."""

    def test_rigid_force_grads_vs_fd(self) -> None:
        num_steps = 4
        forces = 0.5 * np.sin(np.arange(6 * num_steps, dtype=np.float64)).reshape(
            6, num_steps
        )

        def build():
            scene, cube = scenes.rigid_free()
            configure_for_differentiability(scene)
            return scene, cube

        dofs = np.arange(6, dtype=np.int32)

        def apply_factory(cube):
            def apply_inputs(step: int) -> None:
                cube.set_external_forces_on_dofs(
                    dofs, np.ascontiguousarray(forces[:, step])
                )

            return apply_inputs

        scene, cube = build()
        self.addCleanup(physics.destroy_scene, scene)
        loss = TranslationErrorLoss(cube)
        rollout = DifferentiableRollout(scene, dt=DT, num_steps=num_steps)
        result = rollout.run(
            apply_inputs=apply_factory(cube), terminal_losses=[loss]
        )
        adjoint = result.gradients[cube.get_name()].external_forces
        self.assertEqual(adjoint.shape, (6, num_steps))
        self.assertTrue(result.fd_valid)

        def rollout_loss(perturbed: np.ndarray) -> float:
            fd_scene, fd_cube = build()
            try:
                fd_loss = TranslationErrorLoss(fd_cube)
                for step in range(num_steps):
                    fd_cube.set_external_forces_on_dofs(
                        dofs, np.ascontiguousarray(perturbed[:, step])
                    )
                    fd_scene.step(DT)
                return fd_loss.value()
            finally:
                physics.destroy_scene(fd_scene)

        eps = 1e-5
        fd = np.zeros_like(adjoint)
        for d in range(6):
            for step in range(num_steps):
                plus = forces.copy()
                plus[d, step] += eps
                minus = forces.copy()
                minus[d, step] -= eps
                fd[d, step] = (rollout_loss(plus) - rollout_loss(minus)) / (2 * eps)

        self.assertTrue(np.any(np.abs(fd) > 0.0), "test is vacuous")
        np.testing.assert_allclose(adjoint, fd, rtol=1e-6, atol=1e-12)


@double_only
class SoftRolloutTest(unittest.TestCase):
    """The driver on standalone soft actors (phase 5b/5c)."""

    def test_matches_manual_sweep_on_contact_scene(self) -> None:
        num_steps = 4

        def build():
            scene, cube = scenes.soft_cube_on_plane("rich")
            configure_for_differentiability(scene)
            return scene, cube

        # Hand-written sweep with the raw per-step API.
        scene, cube = build()
        self.addCleanup(physics.destroy_scene, scene)
        loss = DisplacementErrorLoss(cube)
        num_dofs = cube.get_num_dofs()
        pre, post = [], []
        for _ in range(num_steps):
            pre.append(scene.capture_state())
            scene.step(DT)
            post.append(scene.capture_state())
        manual_loss = loss.value()
        diffsim.reset_back_propagation(scene)
        for i in range(num_steps, 0, -1):
            diffsim.prepare_back_propagate(scene, post[i - 1], pre[i - 1])
            if i == num_steps:
                loss.accumulate_output_grad()
            diffsim.back_propagate(scene)
        manual_u0 = np.zeros(num_dofs)
        diffsim.set_displacements_backward(cube, manual_u0)
        manual_v0 = np.zeros(num_dofs)
        diffsim.set_node_velocities_local_backward(cube, manual_v0)
        scene.release_all_states()

        # The driver on a fresh, identical scene.
        scene2, cube2 = build()
        self.addCleanup(physics.destroy_scene, scene2)
        rollout = DifferentiableRollout(scene2, dt=DT, num_steps=num_steps)
        result = rollout.run(terminal_losses=[DisplacementErrorLoss(cube2)])
        grads = result.gradients[cube2.get_name()]
        self.assertTrue(result.fd_valid)
        self.assertEqual(result.steps_swept, num_steps)
        self.assertEqual(result.loss, manual_loss)
        self.assertGreater(np.abs(manual_u0).max(), 0.0, "test is vacuous")
        np.testing.assert_array_equal(grads.initial_pose, manual_u0)
        np.testing.assert_array_equal(grads.initial_velocity, manual_v0)
        self.assertIsNone(grads.control_targets)
        self.assertIsNone(grads.external_forces)
        self.assertEqual(grads.force_dofs, [])

    def test_running_loss_vs_fd(self) -> None:
        num_steps = 3

        def build():
            scene, cube = scenes.soft_cube(squash=0.05)
            configure_for_differentiability(scene)
            return scene, cube

        scene, cube = build()
        self.addCleanup(physics.destroy_scene, scene)
        num_dofs = cube.get_num_dofs()
        u0 = np.array(cube.get_displacements())
        v0 = np.tile(np.array([0.3, 0.0, 0.0]), num_dofs // 3)
        rollout = DifferentiableRollout(scene, dt=DT, num_steps=num_steps)
        result = rollout.run(step_losses=lambda _step: [DisplacementErrorLoss(cube)])
        adjoint = result.gradients[cube.get_name()].initial_pose
        self.assertTrue(result.fd_valid)

        def rollout_loss(u_init: np.ndarray) -> float:
            fd_scene, fd_cube = build()
            try:
                fd_cube.set_displacements(u_init)
                fd_cube.set_node_velocities_local(v0)
                fd_loss = DisplacementErrorLoss(fd_cube)
                total = 0.0
                for _ in range(num_steps):
                    fd_scene.step(DT)
                    total += fd_loss.value()
                return total
            finally:
                physics.destroy_scene(fd_scene)

        eps = 1e-6
        fd = np.zeros(num_dofs)
        for i in range(num_dofs):
            plus, minus = u0.copy(), u0.copy()
            plus[i] += eps
            minus[i] -= eps
            fd[i] = (rollout_loss(plus) - rollout_loss(minus)) / (2 * eps)
        self.assertGreater(np.abs(fd).max(), 0.0, "test is vacuous")
        # Same scene and tolerance as SoftRunningLossTest in test_diffsim_soft.
        np.testing.assert_allclose(adjoint, fd, rtol=1e-6, atol=1e-12)

    def test_truncation_withholds_initial_state(self) -> None:
        scene, cube = scenes.soft_cube_on_plane("rich")
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        rollout = DifferentiableRollout(scene, dt=DT, num_steps=3, truncation_window=1)
        result = rollout.run(terminal_losses=[DisplacementErrorLoss(cube)])
        self.assertEqual(result.steps_swept, 1)
        grads = result.gradients[cube.get_name()]
        self.assertIsNone(grads.initial_pose)
        self.assertIsNone(grads.initial_velocity)


def scene_full_actor(scene):
    """The single dynamic articulated actor of a pendulum scene."""
    actors = []
    scene.for_each_actor(actors.append)
    for actor in actors:
        if not actor.is_static() and not actor.is_nested_link_actor():
            return actor
    raise AssertionError("no dynamic actor found")


if __name__ == "__main__":
    unittest.main()


@double_only
class SubstepTest(unittest.TestCase):
    """Failure-adaptive substepping with an injected failure predicate.

    The predicate is keyed on the attempt index, so the split schedule is fixed
    and the finite-difference reference replays the very same schedule by hand:
    step 2 -> two halves; step 4 -> the first half fails again -> two quarters
    plus one half.
    """

    FD_EPS = 1e-6
    TOL = 1e-5
    SCHEDULE = {2: [DT / 2, DT / 2], 4: [DT / 4, DT / 4, DT / 2]}
    # Attempt indices (one per scene.step in the forward rollout) declared failed:
    # step 2 at DT; step 4 at DT and its first half.
    FAILED_ATTEMPTS = {2, 6, 7}

    def _inject_failures(self, failed_attempts) -> list:
        attempts: list[float] = []
        original = diffsim_rollout.forward_solve_failed

        def predicate(scene, residual_tolerance) -> bool:
            attempts.append(residual_tolerance)
            return (len(attempts) - 1) in failed_attempts

        diffsim_rollout.forward_solve_failed = predicate
        self.addCleanup(setattr, diffsim_rollout, "forward_solve_failed", original)
        return attempts

    def test_split_schedule_and_gradients_vs_fd(self) -> None:
        scene, chain, targets, forces, apply_inputs = _controller_setup()
        self.addCleanup(physics.destroy_scene, scene)
        num_dofs = chain.get_num_dofs()
        pose0 = targets[:, 0].copy()
        terminal = ArticulatedPoseErrorLoss(chain, ref=pose0 + 0.1)
        running = ArticulatedPoseErrorLoss(chain, ref=pose0 - 0.05)
        state_init = scene.capture_state()
        attempts = self._inject_failures(self.FAILED_ATTEMPTS)

        rollout = DifferentiableRollout(
            scene,
            dt=DT,
            num_steps=NUM_STEPS,
            max_substep_levels=2,
            substep_residual_tolerance=1e-7,
        )
        result = rollout.run(
            apply_inputs=apply_inputs,
            terminal_losses=[terminal],
            step_losses=lambda step: [running],
        )
        self.assertEqual(len(attempts), NUM_STEPS + 2 + 4)
        self.assertTrue(all(tol == 1e-7 for tol in attempts))
        self.assertEqual(result.split_steps, [(2, 2), (4, 3)])
        self.assertEqual(result.num_solver_steps, NUM_STEPS + 3)
        self.assertEqual(result.steps_swept, NUM_STEPS + 3)
        grads = result.gradients["chain"]

        def objective() -> float:
            scene.restore_state(state_init, False)
            total = 0.0
            for step in range(NUM_STEPS):
                apply_inputs(step)
                for sub_dt in self.SCHEDULE.get(step, [DT]):
                    scene.step(sub_dt)
                total += running.value()
            return total + terminal.value()

        self.assertAlmostEqual(objective(), result.loss, delta=1e-12 * max(1.0, abs(result.loss)))

        def fd_block(array: np.ndarray) -> np.ndarray:
            fd = np.zeros_like(array)
            for index in np.ndindex(array.shape):
                values = []
                for sign in (+1.0, -1.0):
                    saved = array[index]
                    array[index] = saved + sign * self.FD_EPS
                    values.append(objective())
                    array[index] = saved
                fd[index] = (values[0] - values[1]) / (2.0 * self.FD_EPS)
            return fd

        fd_controls = fd_block(targets)
        fd_forces = fd_block(forces)
        scene.release_all_states()
        for name, analytic, fd in (
            ("controls", grads.control_targets, fd_controls),
            ("forces", grads.external_forces, fd_forces),
        ):
            rel = np.linalg.norm(analytic - fd) / np.linalg.norm(fd)
            self.assertLessEqual(rel, self.TOL, f"{name} gradient mismatch:\n{analytic}\n{fd}")
            # The split steps must carry the summed substep gradients, not the last one.
            for step, _ in result.split_steps:
                self.assertGreater(np.linalg.norm(fd[:, step]), 0.0)
        self.assertEqual(num_dofs, grads.control_targets.shape[0])

    def test_finest_level_failure_raises(self) -> None:
        scene, chain, targets, forces, apply_inputs = _controller_setup()
        self.addCleanup(physics.destroy_scene, scene)
        terminal = ArticulatedPoseErrorLoss(chain, ref=targets[:, 0] + 0.1)
        # Attempts: step 0 ok; step 1 fails at DT, DT/2 and DT/4 -> error at DT/4.
        self._inject_failures({1, 2, 3})
        rollout = DifferentiableRollout(
            scene, dt=DT, num_steps=NUM_STEPS, max_substep_levels=2, substep_residual_tolerance=1e-7
        )
        with self.assertRaises(ForwardSolveError) as ctx:
            rollout.run(apply_inputs=apply_inputs, terminal_losses=[terminal])
        self.assertEqual(ctx.exception.step, 1)
        self.assertAlmostEqual(ctx.exception.dt, DT / 4)
        # All captured states were released on the error path.
        scene.release_all_states()

    def test_argument_validation(self) -> None:
        scene, _ = scenes.rigid_free()
        self.addCleanup(physics.destroy_scene, scene)
        with self.assertRaises(ValueError):
            DifferentiableRollout(scene, dt=DT, num_steps=2, max_substep_levels=-1)
        with self.assertRaises(ValueError):
            DifferentiableRollout(scene, dt=DT, num_steps=2, max_substep_levels=1)
        with self.assertRaises(ValueError):
            DifferentiableRollout(
                scene, dt=DT, num_steps=2, max_substep_levels=1, substep_residual_tolerance=0.0
            )
        with self.assertRaises(ValueError):
            DifferentiableRollout(scene, dt=DT, num_steps=2, substep_residual_tolerance=1e-6)
        with self.assertRaises(ValueError):
            diffsim_rollout.step_with_substeps(scene, DT, max_levels=1, residual_tolerance=-1.0)

    def test_without_substepping_convergence_is_not_inspected(self) -> None:
        scene, chain, targets, forces, apply_inputs = _controller_setup()
        self.addCleanup(physics.destroy_scene, scene)
        terminal = ArticulatedPoseErrorLoss(chain, ref=targets[:, 0] + 0.1)
        attempts = self._inject_failures(set(range(100)))
        result = DifferentiableRollout(scene, dt=DT, num_steps=NUM_STEPS).run(
            apply_inputs=apply_inputs, terminal_losses=[terminal]
        )
        self.assertEqual(attempts, [])
        self.assertEqual(result.split_steps, [])
        self.assertEqual(result.num_solver_steps, NUM_STEPS)


class _Stats:
    """A stand-in for the engine's solver statistics."""

    def __init__(self, status, residual: float, iters: int = 7):
        self.convergence_status = status
        self.residual_norm = residual
        self.max_non_linear_iters = iters


class ForwardFailurePredicateTest(unittest.TestCase):
    """``forward_solve_failed`` on every solver status and every class of residual: finite
    residuals follow the status-and-threshold rule, a residual that is not a finite number
    is a failure whatever the status (before 2026-09-08 ``nan > tolerance`` being false let
    a NaN residual pass as converged)."""

    STATUSES = [s for n, s in physics.ConvergenceStatus.__members__.items() if n != "COUNT"]
    TOL = 1e-9

    def _predicate(self, status, residual: float) -> bool:
        with mock.patch.object(diffsim_rollout, "_solver_stats", lambda scene: _Stats(status, residual)):
            return diffsim_rollout.forward_solve_failed(None, self.TOL)

    def test_finite_residuals_follow_the_status_and_threshold(self) -> None:
        for status in self.STATUSES:
            for residual in (0.0, 1e-12, 1e-9, 2e-9, 1e-6, 1.0):
                expected = status != physics.ConvergenceStatus.CONVERGED and residual > self.TOL
                with self.subTest(status=status, residual=residual):
                    self.assertEqual(self._predicate(status, residual), expected)

    def test_non_finite_residuals_fail_whatever_the_status(self) -> None:
        for status in self.STATUSES:
            for residual in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(status=status, residual=residual):
                    self.assertTrue(self._predicate(status, residual))


@double_only
class NonFiniteResidualTest(unittest.TestCase):
    """A step whose Newton residual is NaN is an error in every mode: with substepping it
    is subdivided and the finest level raises; without substepping (where convergence is
    otherwise not inspected) it raises at once. Either way every capture is released."""

    def _nan_stats(self, real, actual):
        return _Stats(physics.ConvergenceStatus.STOPPED, float("nan"), actual.max_non_linear_iters)

    def _nan_from_second_attempt(self) -> list:
        """Counts the failure predicate's calls (one per solve attempt) and makes the solver
        statistics NaN from the second attempt on, i.e. for step 1 and its substeps."""
        real_stats = diffsim_rollout._solver_stats
        real_failed = diffsim_rollout.forward_solve_failed
        attempts: list[float] = []

        def stats(scene):
            actual = real_stats(scene)
            return self._nan_stats(real_stats, actual) if len(attempts) >= 2 else actual

        def failed(scene, residual_tolerance) -> bool:
            attempts.append(residual_tolerance)
            return real_failed(scene, residual_tolerance)

        for name, replacement in (("_solver_stats", stats), ("forward_solve_failed", failed)):
            patcher = mock.patch.object(diffsim_rollout, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        return attempts

    def _nan_from_second_step(self) -> list:
        """Without substepping the statistics are read once per step: NaN from step 1 on."""
        real_stats = diffsim_rollout._solver_stats
        calls: list[int] = []

        def stats(scene):
            calls.append(len(calls))
            actual = real_stats(scene)
            return self._nan_stats(real_stats, actual) if len(calls) >= 2 else actual

        patcher = mock.patch.object(diffsim_rollout, "_solver_stats", stats)
        patcher.start()
        self.addCleanup(patcher.stop)
        return calls

    def test_substepping_subdivides_then_raises(self) -> None:
        scene, chain, targets, forces, apply_inputs = _controller_setup()
        self.addCleanup(physics.destroy_scene, scene)
        terminal = ArticulatedPoseErrorLoss(chain, ref=targets[:, 0] + 0.1)
        attempts = self._nan_from_second_attempt()
        rollout = DifferentiableRollout(
            scene, dt=DT, num_steps=NUM_STEPS, max_substep_levels=2, substep_residual_tolerance=1e-7
        )
        with self.assertRaises(ForwardSolveError) as ctx:
            rollout.run(apply_inputs=apply_inputs, terminal_losses=[terminal])
        self.assertEqual(ctx.exception.step, 1)
        self.assertAlmostEqual(ctx.exception.dt, DT / 4)
        self.assertTrue(np.isnan(ctx.exception.residual_norm))
        self.assertEqual(len(attempts), 4)  # step 0, then step 1 at DT, DT/2 and DT/4
        scene.release_all_states()

    def test_without_substepping_raises_at_once(self) -> None:
        scene, chain, targets, forces, apply_inputs = _controller_setup()
        self.addCleanup(physics.destroy_scene, scene)
        terminal = ArticulatedPoseErrorLoss(chain, ref=targets[:, 0] + 0.1)
        calls = self._nan_from_second_step()
        with self.assertRaises(ForwardSolveError) as ctx:
            DifferentiableRollout(scene, dt=DT, num_steps=NUM_STEPS).run(
                apply_inputs=apply_inputs, terminal_losses=[terminal]
            )
        self.assertEqual(ctx.exception.step, 1)
        self.assertAlmostEqual(ctx.exception.dt, DT)
        self.assertTrue(np.isnan(ctx.exception.residual_norm))
        self.assertEqual(len(calls), 2)
        scene.release_all_states()


@double_only
class VariableStepSizeTest(unittest.TestCase):
    """The per-step adjoint chained across steps of different dt."""

    FD_EPS = 1e-6
    TOL = 1e-5
    DTS = [DT, DT / 2, DT / 4, DT, DT / 2, DT]

    def _sweep(self, scene, dts, apply_inputs, terminal, read_step, read_initial):
        pre, post = [], []
        for step, dt in enumerate(dts):
            apply_inputs(step)
            pre.append(scene.capture_state())
            scene.step(dt)
            post.append(scene.capture_state())
        loss = terminal.value()
        step_grads = []
        diffsim.reset_back_propagation(scene)
        for i in range(len(dts) - 1, -1, -1):
            diffsim.prepare_back_propagate(scene, post[i], pre[i])
            if i == len(dts) - 1:
                terminal.accumulate_output_grad()
            diffsim.back_propagate(scene)
            step_grads.append(read_step())
        initial = read_initial()
        for handle in pre + post:
            scene.release_state(handle)
        return loss, step_grads[::-1], initial

    def _per_step_errors(self, scene, chain, inputs, apply_inputs, read_step):
        """Relative error per step of the analytic per-step input gradient vs central FD
        of ``inputs`` (an array (dofs, steps) that ``apply_inputs`` reads)."""
        num_dofs = chain.get_num_dofs()
        pose0 = np.zeros(num_dofs)
        chain.get_articulated_pose(pose0)
        terminal = ArticulatedPoseErrorLoss(chain, ref=pose0 + 0.1)
        state_init = scene.capture_state()
        loss, step_grads, _ = self._sweep(
            scene, self.DTS, apply_inputs, terminal, read_step, lambda: None
        )
        analytic = np.stack(step_grads, axis=1)

        def objective() -> float:
            scene.restore_state(state_init, False)
            for step, dt in enumerate(self.DTS):
                apply_inputs(step)
                scene.step(dt)
            return terminal.value()

        self.assertAlmostEqual(objective(), loss, delta=1e-12 * max(1.0, abs(loss)))
        fd = np.zeros_like(inputs)
        for index in np.ndindex(inputs.shape):
            values = []
            for sign in (+1.0, -1.0):
                saved = inputs[index]
                inputs[index] = saved + sign * self.FD_EPS
                values.append(objective())
                inputs[index] = saved
            fd[index] = (values[0] - values[1]) / (2.0 * self.FD_EPS)
        scene.release_all_states()
        return [
            np.linalg.norm(analytic[:, step] - fd[:, step]) / np.linalg.norm(fd[:, step])
            for step in range(NUM_STEPS)
        ]

    def test_articulated_joint_forces_vs_fd(self) -> None:
        scene, chain = scenes.pendulum(with_controller=False)
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        num_dofs = chain.get_num_dofs()
        force_dofs = np.arange(num_dofs, dtype=np.int32)
        forces = np.stack([DT * (j + 1) * np.ones(num_dofs) for j in range(NUM_STEPS)], axis=1)

        def apply_inputs(step: int) -> None:
            chain.set_external_forces_on_dofs(force_dofs, np.ascontiguousarray(forces[:, step]))

        def read_step():
            g = np.zeros(num_dofs)
            diffsim.set_external_forces_on_dofs_backward(chain, force_dofs, g)
            return g

        errors = self._per_step_errors(scene, chain, forces, apply_inputs, read_step)
        self.assertLessEqual(max(errors), self.TOL, f"joint-force gradient errors per step: {errors}")

    def test_articulated_controller_targets_vs_fd(self) -> None:
        """Pose-controller target gradients at variable dt (with the controller's
        damping term, which couples consecutive targets through the target velocity).

        Before the engine re-expressed finite-difference angular velocities for a
        changed step size, these were off by ~1e-4..1e-3 (up to 3e-2 at 0.2 rad per
        step): the rotation predictor of the previous joint increment was a linearly
        scaled matrix, not a rotation, while the adjoint's delta chain assumes one.
        """
        scene, chain, targets, forces, apply_inputs = _controller_setup()
        self.addCleanup(physics.destroy_scene, scene)
        num_dofs = chain.get_num_dofs()

        def apply_targets(step: int) -> None:
            if step == 0:
                chain.set_articulated_target_velocity(np.zeros(num_dofs))
            chain.set_articulated_target_pose(np.ascontiguousarray(targets[:, step]))

        def read_step():
            g = np.zeros(num_dofs)
            diffsim.set_articulated_target_pose_backward(chain, g)
            return g

        errors = self._per_step_errors(scene, chain, targets, apply_targets, read_step)
        self.assertLessEqual(max(errors), self.TOL, f"control gradient errors per step: {errors}")

    def test_spin_rate_preserved_across_dt_change(self) -> None:
        """Forward semantics of a step-size change. The rigid-body merit extrapolates the
        previous rotation increment DR linearly, R~ = (2 I - DR^T) R, and a free spin (no
        gravity, no torque, isotropic inertia) settles on the rotation nearest to R~: an
        increment of phi(theta) = atan(sin theta / (2 - cos theta)) for a previous
        increment theta (the extrapolation's own per-step map, 0.2 rad -> 0.1936 rad).
        When the step shrinks by s, the engine re-encodes the previous finite-difference
        angular velocity so that the extrapolated increment is s theta (the angular rate
        is preserved), hence theta2 = phi(s theta1). The old matrix-scaled extrapolation
        (not a rotation) gave atan(s sin theta / (1 + s (1 - cos theta))), 6e-3 relative
        away here."""
        scene, cube = scenes.rigid_free()
        self.addCleanup(physics.destroy_scene, scene)
        scene.set_gravity([0.0, 0.0, 0.0])
        configure_for_differentiability(scene)
        cube.set_velocity(np.zeros(3), np.array([20.0, 0.0, 0.0]))  # 0.2 rad per 10 ms step

        def rotation_angle(before, after) -> float:
            # angle of the relative rotation after * before^-1 from the quaternions (x, y, z, w)
            qb = np.array([before.rotation[i] for i in range(4)])
            qa = np.array([after.rotation[i] for i in range(4)])
            # q_rel = qa * conj(qb); w component:
            w = qa[3] * qb[3] + qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2]
            return 2.0 * np.arccos(min(1.0, abs(w)))

        t0 = cube.get_center_of_mass_transform()
        scene.step(DT)
        t1 = cube.get_center_of_mass_transform()
        scene.step(DT / 2)
        t2 = cube.get_center_of_mass_transform()
        theta1 = rotation_angle(t0, t1)
        theta2 = rotation_angle(t1, t2)
        self.assertGreater(theta1, 0.15)
        scaled = 0.5 * theta1
        expected = np.arctan2(np.sin(scaled), 2.0 - np.cos(scaled))
        self.assertLess(abs(theta2 - expected), 1e-9 * theta1, f"theta1 {theta1}, theta2 {theta2}, expected {expected}")

    def test_rigid_initial_velocity_vs_fd(self) -> None:
        scene, cube = scenes.rigid_free()
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        terminal = TranslationErrorLoss(cube)
        state_init = scene.capture_state()

        def read_initial():
            gl = np.zeros(3)
            ga = np.zeros(3)
            diffsim.set_velocity_backward(cube, gl, ga)
            return np.concatenate([gl, ga])

        loss, _, analytic = self._sweep(
            scene, self.DTS, lambda step: None, terminal, lambda: None, read_initial
        )

        def objective() -> float:
            for dt in self.DTS:
                scene.step(dt)
            return terminal.value()

        fd = np.zeros(6)
        for j in range(6):
            values = []
            for sign in (+1.0, -1.0):
                scene.restore_state(state_init, False)
                lin = np.asarray(cube.get_linear_velocity(), dtype=np.float64)
                ang = np.asarray(cube.get_angular_velocity(), dtype=np.float64)
                delta = np.zeros(3)
                delta[j % 3] = sign * self.FD_EPS
                if j < 3:
                    lin = lin + delta
                else:
                    ang = ang + delta
                cube.set_velocity(lin, ang)
                values.append(objective())
            fd[j] = (values[0] - values[1]) / (2.0 * self.FD_EPS)
        scene.release_all_states()
        rel = np.linalg.norm(analytic - fd) / np.linalg.norm(fd)
        self.assertLessEqual(rel, self.TOL, f"initial-velocity gradient mismatch: {analytic} vs {fd}")


def _live_handles(scene, handles) -> list:
    """The handles among ``handles`` the scene still holds (a released one raises on restore)."""
    live = []
    for handle in handles:
        try:
            scene.restore_state(handle, False)
        except RuntimeError:
            continue
        live.append(handle)
    return live


class _RaisingLoss:
    def __init__(self, where: str):
        self.where = where

    def value(self) -> float:
        if self.where == "value":
            raise RuntimeError("intentional loss error")
        return 0.0

    def accumulate_output_grad(self) -> None:
        if self.where == "grad":
            raise RuntimeError("intentional gradient error")


class SnapshotLifetimeTest(unittest.TestCase):
    """Every state the driver captures is released when the run fails, whatever raised.

    Until 2026-09-07 the captures were released on the success path only: a loss raising in ``value()`` or ``accumulate_output_grad()``, or the
    adjoint itself, left every step's pre and post state alive - eight handles for four steps -
    and a native error inside ``scene.step`` leaked the step's pre-step capture. Released
    handles raise on restore, which is how these tests count the survivors; the captures are
    collected through the scene's ``capture_state``.
    """

    def _run_failing(self, expected_captures: int, **run_kwargs):
        scene, cube = scenes.rigid_on_plane("none")
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        captured = []
        real_capture = type(scene).capture_state

        def capture(self_scene):
            handle = real_capture(self_scene)
            captured.append(handle)
            return handle

        with mock.patch.object(type(scene), "capture_state", capture):
            with self.assertRaises(RuntimeError):
                DifferentiableRollout(scene, dt=DT, num_steps=4).run(**run_kwargs)
        self.assertEqual(len(captured), expected_captures)
        self.assertEqual(_live_handles(scene, captured), [])

    def test_failing_terminal_loss_value(self) -> None:
        self._run_failing(8, terminal_losses=[_RaisingLoss("value")])

    def test_failing_output_gradient(self) -> None:
        self._run_failing(8, terminal_losses=[_RaisingLoss("grad")])

    def test_failing_step_loss_value(self) -> None:
        # Running costs are evaluated live: the second step's loss raises inside the forward
        # rollout, after the captures of two steps.
        self._run_failing(4, step_losses=lambda step: [_RaisingLoss("value" if step == 1 else "none")])

    def test_failing_step_loss_gradient(self) -> None:
        self._run_failing(8, step_losses=lambda step: [_RaisingLoss("grad" if step == 1 else "none")])

    def test_native_step_error_releases_the_pre_step_capture(self) -> None:
        scene, cube = scenes.rigid_on_plane("none")
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        captured = []
        real_capture = type(scene).capture_state

        def capture(self_scene):
            handle = real_capture(self_scene)
            captured.append(handle)
            return handle

        real_step = type(scene).step
        calls = []

        def step(self_scene, dt):
            calls.append(dt)
            if len(calls) == 3:
                raise RuntimeError("intentional native step error")
            return real_step(self_scene, dt)

        loss = TranslationErrorLoss(cube)
        with mock.patch.object(type(scene), "capture_state", capture), mock.patch.object(
            type(scene), "step", step
        ):
            with self.assertRaises(RuntimeError):
                DifferentiableRollout(scene, dt=DT, num_steps=4).run(terminal_losses=[loss])
        self.assertEqual(len(captured), 5)  # two full steps and the failing step's pre-state
        self.assertEqual(_live_handles(scene, captured), [])


class TruncationObjectiveTest(unittest.TestCase):
    """``truncation_window`` truncates the reverse sweep, not the objective: until 2026-09-07 the
    running costs were summed while sweeping, so a window of 2 over 6 steps of a constant cost
    1 reported 2 instead of 6, and a gradient option changed
    the value being optimized."""

    class _ConstantLoss:
        def value(self) -> float:
            return 1.0

        def accumulate_output_grad(self) -> None:
            pass

    def test_constant_running_cost(self) -> None:
        for window, swept in ((None, NUM_STEPS), (2, 2)):
            scene, cube = scenes.rigid_on_plane("none")
            self.addCleanup(physics.destroy_scene, scene)
            configure_for_differentiability(scene)
            result = DifferentiableRollout(
                scene, dt=DT, num_steps=NUM_STEPS, truncation_window=window
            ).run(step_losses=lambda step: [self._ConstantLoss()])
            self.assertEqual(result.steps_swept, swept)
            self.assertEqual(result.loss, float(NUM_STEPS))

    def test_running_loss_value_is_window_independent(self) -> None:
        losses = []
        for window in (None, 2):
            scene, cube = scenes.rigid_free()
            self.addCleanup(physics.destroy_scene, scene)
            configure_for_differentiability(scene)
            loss = TranslationErrorLoss(cube)
            result = DifferentiableRollout(
                scene, dt=DT, num_steps=NUM_STEPS, truncation_window=window
            ).run(step_losses=lambda step: [loss])
            losses.append(result.loss)
        self.assertGreater(losses[0], 0.0)
        self.assertAlmostEqual(losses[0], losses[1], delta=1e-12 * losses[0])


class _CachedTranslationLoss:
    """A running cost that caches its residual in ``value()`` and differentiates from the
    cache: the two-method protocol allows it, and the driver must call ``value()`` on the
    restored step right before the gradient (before 2026-09-07: the
    driver evaluated the values live only, so a fresh instance per step got its gradient
    requested without a value, and a shared instance differentiated the last forward step's
    residual at every step - 30% off)."""

    def __init__(self, actor, ref=(0.0, 0.0, 0.0)):
        self.actor = actor
        self.ref = np.asarray(ref, dtype=np.float64)
        self.residual = None
        self.values = 0
        self.gradients = 0

    def value(self) -> float:
        self.values += 1
        self.residual = (
            np.asarray(self.actor.get_center_of_mass_transform().translation, dtype=np.float64)
            - self.ref
        )
        return 0.5 * float(self.residual @ self.residual)

    def accumulate_output_grad(self) -> None:
        if self.residual is None:
            raise RuntimeError("gradient requested before loss.value()")
        self.gradients += 1
        grad = np.zeros(7, dtype=_real_dtype())
        grad[:3] = self.residual
        diffsim.get_center_of_mass_transform_backward(self.actor, grad)
        self.residual = None  # a stale cache must never serve a later step


def _real_dtype():
    return np.float64 if DOUBLE else np.float32


class CachedRunningLossTest(unittest.TestCase):
    """Running costs that cache their derivative context in ``value()``, fresh per step and
    shared across steps, against the stateless harness loss and against finite differences."""

    def _free_fall(self):
        scene, cube = scenes.rigid_free()
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        cube.set_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        return scene, cube

    def test_fresh_cached_losses_match_the_stateless_loss(self) -> None:
        scene, cube = self._free_fall()
        stateless = DifferentiableRollout(scene, dt=0.05, num_steps=6).run(
            step_losses=lambda step: [TranslationErrorLoss(cube, ref=np.array([0.1, 0.2, 0.3]))]
        )
        scene.release_all_states()
        scene, cube = self._free_fall()
        made = []

        def losses(step):
            loss = _CachedTranslationLoss(cube, ref=(0.1, 0.2, 0.3))
            made.append(loss)
            return [loss]

        cached = DifferentiableRollout(scene, dt=0.05, num_steps=6).run(step_losses=losses)
        self.assertEqual(cached.loss, stateless.loss)
        np.testing.assert_array_equal(
            cached.gradients["cube"].initial_velocity, stateless.gradients["cube"].initial_velocity
        )
        # Every instance was evaluated before its gradient; the sweep's instances once more.
        self.assertTrue(all(loss.values >= 1 for loss in made))
        self.assertEqual(sum(loss.gradients for loss in made), 6)

    def test_shared_cached_loss_vs_fd(self) -> None:
        n, dt = 6, 0.05
        scene, cube = self._free_fall()
        initial = scene.capture_state()
        dofs = np.arange(6, dtype=np.int32)
        forces = np.zeros((n, 6))
        loss = _CachedTranslationLoss(cube)

        def apply(step):
            cube.set_external_forces_on_dofs(dofs, np.ascontiguousarray(forces[step]))

        result = DifferentiableRollout(scene, dt=dt, num_steps=n).run(
            apply_inputs=apply, step_losses=lambda step: [loss]
        )
        analytic = float(result.gradients["cube"].external_forces[2, 0])
        # The position is linear in the force, so the loss is quadratic in it and the central
        # difference is exact up to rounding: the wide float32 steps keep the rounding of the
        # loss (1e-7 of 1.9) below the tolerance.
        steps, tol = ((1e-4, 1e-5), 1e-6) if DOUBLE else ((1e-1, 5e-2), 5e-3)
        fds = []
        for eps in steps:
            values = []
            for sign in (1.0, -1.0):
                scene.restore_state(initial, False)
                forces[0, 2] = sign * eps
                total = 0.0
                for step in range(n):
                    apply(step)
                    scene.step(dt)
                    total += loss.value()
                values.append(total)
            fds.append((values[0] - values[1]) / (2 * eps))
        forces[0, 2] = 0.0
        scene.release_all_states()
        self.assertGreater(abs(fds[0]), 1e-6, "vacuous")
        self.assertLessEqual(abs(fds[0] - fds[1]) / abs(fds[0]), tol, fds)
        self.assertLessEqual(abs(analytic - fds[0]) / abs(fds[0]), tol, (analytic, fds))


class AdaptiveSnapshotLifetimeTest(unittest.TestCase):
    """The failure-adaptive step helper owns a step's pre-state until both captures reached the
    record list (before 2026-09-07: a failing post-state capture leaked
    the pre-state with substepping enabled, even without an actual subdivision)."""

    def _captures(self, scene):
        captured = []
        real_capture = type(scene).capture_state
        state = {"calls": 0, "fail_at": None}

        def capture(self_scene):
            state["calls"] += 1
            if state["calls"] == state["fail_at"]:
                raise RuntimeError("injected capture failure")
            handle = real_capture(self_scene)
            captured.append(handle)
            return handle

        return captured, state, capture

    def test_post_state_capture_failure(self) -> None:
        scene, cube = scenes.rigid_free()
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        captured, state, capture = self._captures(scene)
        state["fail_at"] = 2  # the post-state of the first step
        loss = TranslationErrorLoss(cube)
        # The convergence check is not under test (the float32 build's residuals would not pass
        # the tight tolerance the adaptive mode requires): the step is accepted.
        with mock.patch.object(type(scene), "capture_state", capture), mock.patch.object(
            diffsim_rollout, "forward_solve_failed", return_value=False
        ):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                DifferentiableRollout(
                    scene, dt=DT, num_steps=2, max_substep_levels=1, substep_residual_tolerance=1e-6
                ).run(terminal_losses=[loss])
        self.assertEqual(len(captured), 1)
        self.assertEqual(_live_handles(scene, captured), [])

    def test_capture_failure_after_an_accepted_half_step(self) -> None:
        scene, cube = scenes.rigid_free()
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        captured, state, capture = self._captures(scene)
        # The plain step is declared failed once, so it is redone as two half steps: captures
        # are the pre-state (1), the first half's post-state (2, accepted into the records),
        # the second half's pre-state (3) and its post-state (4), which fails.
        state["fail_at"] = 4
        loss = TranslationErrorLoss(cube)
        failed = mock.Mock(side_effect=[True, False, False])
        with mock.patch.object(type(scene), "capture_state", capture), mock.patch.object(
            diffsim_rollout, "forward_solve_failed", failed
        ):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                DifferentiableRollout(
                    scene, dt=DT, num_steps=2, max_substep_levels=1, substep_residual_tolerance=1e-6
                ).run(terminal_losses=[loss])
        self.assertEqual(failed.call_count, 3)
        self.assertEqual(len(captured), 3)
        self.assertEqual(_live_handles(scene, captured), [])

    def test_failure_in_the_convergence_check_or_the_restore(self) -> None:
        """The other failure paths between the step and the hand-over: a raising convergence
        check, and a raising restore before a subdivision."""
        for where in ("check", "restore"):
            scene, cube = scenes.rigid_free()
            self.addCleanup(physics.destroy_scene, scene)
            configure_for_differentiability(scene)
            captured, state, capture = self._captures(scene)
            loss = TranslationErrorLoss(cube)
            if where == "check":
                failed = mock.Mock(side_effect=RuntimeError("injected check failure"))
                restore = type(scene).restore_state
            else:
                failed = mock.Mock(return_value=True)
                restore = mock.Mock(side_effect=RuntimeError("injected restore failure"))
            with mock.patch.object(type(scene), "capture_state", capture), mock.patch.object(
                diffsim_rollout, "forward_solve_failed", failed
            ), mock.patch.object(type(scene), "restore_state", restore):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    DifferentiableRollout(
                        scene, dt=DT, num_steps=2, max_substep_levels=1, substep_residual_tolerance=1e-6
                    ).run(terminal_losses=[loss])
            self.assertEqual(len(captured), 1, where)
            self.assertEqual(_live_handles(scene, captured), [], where)


class LossFactoryReplayTest(unittest.TestCase):
    """``step_losses(step)`` is called exactly once per step, during the forward rollout, and
    the sweep differentiates the instances it returned (before 2026-09-07: the factory was called again in the sweep, so a factory that samples or consumes
    data - targets, weights, minibatches - had its value and its gradient taken from two
    different objectives, 2.9x apart in one measured case, and an iterator-backed factory
    ran dry). The finite-difference references below hold the targets the forward rollout
    actually drew."""

    N, DT_FREE = 6, 0.05

    def _free_fall(self):
        scene, cube = scenes.rigid_free()
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        cube.set_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        return scene, cube

    def _check_factory(self, make_factory) -> None:
        scene, cube = self._free_fall()
        initial = scene.capture_state()
        dofs = np.arange(6, dtype=np.int32)
        forces = np.zeros((self.N, 6))
        calls, refs = [], []
        factory = make_factory(cube, calls, refs)

        def apply(step):
            cube.set_external_forces_on_dofs(dofs, np.ascontiguousarray(forces[step]))

        result = DifferentiableRollout(scene, dt=self.DT_FREE, num_steps=self.N).run(
            apply_inputs=apply, step_losses=factory
        )
        self.assertEqual(calls, list(range(self.N)))  # once per step, in order
        analytic = float(result.gradients["cube"].external_forces[2, 0])

        def objective():
            total = 0.0
            for step in range(self.N):
                apply(step)
                scene.step(self.DT_FREE)
                total += TranslationErrorLoss(cube, ref=refs[step]).value()
            return total

        steps, tol = ((1e-4, 1e-5), 1e-6) if DOUBLE else ((1e-1, 5e-2), 5e-3)
        fds = []
        for eps in steps:
            values = []
            for sign in (1.0, -1.0):
                scene.restore_state(initial, False)
                forces[0, 2] = sign * eps
                values.append(objective())
            fds.append((values[0] - values[1]) / (2 * eps))
        forces[0, 2] = 0.0
        scene.restore_state(initial, False)
        recorded = objective()
        scene.release_all_states()
        self.assertAlmostEqual(result.loss, recorded, delta=1e-12 * max(1.0, abs(recorded)))
        self.assertLessEqual(abs(fds[0] - fds[1]) / abs(fds[0]), tol, fds)
        self.assertLessEqual(abs(analytic - fds[0]) / abs(fds[0]), tol, (analytic, fds))

    def test_sampling_factory(self) -> None:
        def make(cube, calls, refs):
            rng = np.random.default_rng(42)

            def factory(step):
                target = rng.uniform(0.0, 2.0, 3)
                calls.append(step)
                refs.append(target.copy())
                return [TranslationErrorLoss(cube, ref=target)]

            return factory

        self._check_factory(make)

    def test_iterator_backed_factory(self) -> None:
        def make(cube, calls, refs):
            targets = iter(np.random.default_rng(7).uniform(0.0, 2.0, (self.N, 3)))

            def factory(step):
                target = next(targets)  # a second pass would run dry
                calls.append(step)
                refs.append(target.copy())
                return [TranslationErrorLoss(cube, ref=target)]

            return factory

        self._check_factory(make)

    def test_truncated_sweep_and_adaptive_substeps_call_the_factory_once_per_step(self) -> None:
        reference = None
        for window, adaptive in ((None, False), (2, False), (None, True)):
            scene, cube = self._free_fall()
            calls = []

            def factory(step):
                calls.append(step)
                return [TranslationErrorLoss(cube, ref=np.array([0.1, 0.2, 0.3]))]

            kwargs = dict(max_substep_levels=1, substep_residual_tolerance=1e-6) if adaptive else {}
            if adaptive:
                # The first step is declared failed once and redone as two half steps.
                patch = mock.patch.object(
                    diffsim_rollout, "forward_solve_failed", side_effect=[True] + [False] * 20
                )
            else:
                patch = mock.patch.object(
                    diffsim_rollout, "forward_solve_failed", wraps=diffsim_rollout.forward_solve_failed
                )
            with patch:
                result = DifferentiableRollout(
                    scene, dt=self.DT_FREE, num_steps=self.N, truncation_window=window, **kwargs
                ).run(step_losses=factory)
            self.assertEqual(calls, list(range(self.N)), (window, adaptive))
            if adaptive:
                # The split changes the trajectory (Backward Euler at another step size), not
                # the number of factory calls.
                self.assertEqual(result.split_steps, [(0, 2)])
            elif window is None:
                reference = result.loss
            else:
                self.assertEqual(result.loss, reference, (window, adaptive))


@double_only
class ArticulatedInitialPoseTest(unittest.TestCase):
    """The driver's initial-pose gradient of a torque-driven articulated actor against
    central finite differences (joint velocities held). The links' previous deltas are
    J(q) v dt: the velocity setter derives the link velocities from the joint velocities
    at the pose, and in a differentiable scene the pose setters do the same, so the
    initial state is consistent whatever the setter order and the pose input adjoint
    carries the Jacobian term. Without it a prismatic joint after a revolute one was off
    by about a tenth (the prismatic axis turns with its parent) and a revolute pendulum
    by 1e-4. The protocols: pose then velocities, velocities then pose, and the pose alone
    on a state that has stepped."""

    FD_EPS = 1e-6
    TOL = 1e-4
    NUM_STEPS = 6

    def _relative_error(self, scene, chain, protocol: str = "pose_then_velocities") -> float:
        configure_for_differentiability(scene)
        num_dofs = chain.get_num_dofs()
        torque = 0.3 * np.arange(1, num_dofs + 1, dtype=np.float64)
        dofs = np.arange(num_dofs, dtype=np.int32)

        def apply_inputs(step: int) -> None:
            chain.set_external_forces_on_dofs(dofs, torque)

        if protocol == "pose_alone":
            # A state that has stepped: the pose setter alone re-derives the link
            # velocities from the joint velocities the steps produced.
            for _ in range(3):
                apply_inputs(0)
                scene.step(DT)
        pose0 = np.zeros(num_dofs)
        chain.get_articulated_pose(pose0)
        velocity0 = np.zeros(num_dofs)
        chain.get_articulated_joint_velocities(velocity0)
        terminal = ArticulatedPoseErrorLoss(chain, ref=pose0 + 0.1)

        def set_state(pose: np.ndarray) -> None:
            if protocol == "pose_then_velocities":
                chain.set_articulated_pose_from_joints(pose)
                chain.set_articulated_joint_velocities(velocity0)
            elif protocol == "velocities_then_pose":
                chain.set_articulated_joint_velocities(velocity0)
                chain.set_articulated_pose_from_joints(pose)
            else:
                chain.set_articulated_pose_from_joints(pose)

        # The initial state goes through the setters the gradient is taken with respect to
        # (the pose setter also owns the controller targets it resets).
        set_state(pose0)
        state_init = scene.capture_state()
        rollout = DifferentiableRollout(scene, dt=DT, num_steps=self.NUM_STEPS)
        result = rollout.run(apply_inputs=apply_inputs, terminal_losses=[terminal])
        analytic = result.gradients[chain.get_name()].initial_pose
        self.assertTrue(result.fd_valid)

        def objective(pose: np.ndarray) -> float:
            scene.restore_state(state_init, False)
            set_state(pose)
            for _ in range(self.NUM_STEPS):
                apply_inputs(0)
                scene.step(DT)
            return terminal.value()

        fd = np.zeros(num_dofs)
        for i in range(num_dofs):
            values = []
            for sign in (+1.0, -1.0):
                pose = pose0.copy()
                pose[i] += sign * self.FD_EPS
                values.append(objective(pose))
            fd[i] = (values[0] - values[1]) / (2.0 * self.FD_EPS)
        scene.release_all_states()
        self.assertGreater(np.linalg.norm(fd), 0.0, "test is vacuous")
        return float(np.linalg.norm(analytic - fd) / np.linalg.norm(fd))

    def test_pendulum_initial_pose_vs_fd(self) -> None:
        scene, chain = scenes.pendulum(with_controller=False)
        self.addCleanup(physics.destroy_scene, scene)
        rel = self._relative_error(scene, chain)
        self.assertLessEqual(rel, self.TOL, f"initial-pose gradient mismatch: {rel:.2e}")

    def test_pendulum_with_controller_initial_pose_vs_fd(self) -> None:
        """With a pose controller the pose setter also resets both controller targets, which
        the adjoint carries through the current- and previous-target paths."""
        scene, chain = scenes.pendulum(with_controller=True)
        self.addCleanup(physics.destroy_scene, scene)
        rel = self._relative_error(scene, chain)
        self.assertLessEqual(rel, self.TOL, f"initial-pose gradient mismatch: {rel:.2e}")

    def test_revolute_prismatic_initial_pose_vs_fd(self) -> None:
        scene, chain = scenes.chain_revolute_prismatic()
        self.addCleanup(physics.destroy_scene, scene)
        rel = self._relative_error(scene, chain)
        self.assertLessEqual(rel, self.TOL, f"initial-pose gradient mismatch: {rel:.2e}")

    def test_revolute_prismatic_velocities_then_pose_vs_fd(self) -> None:
        scene, chain = scenes.chain_revolute_prismatic()
        self.addCleanup(physics.destroy_scene, scene)
        rel = self._relative_error(scene, chain, protocol="velocities_then_pose")
        self.assertLessEqual(rel, self.TOL, f"initial-pose gradient mismatch: {rel:.2e}")

    def test_revolute_prismatic_pose_alone_vs_fd(self) -> None:
        scene, chain = scenes.chain_revolute_prismatic()
        self.addCleanup(physics.destroy_scene, scene)
        rel = self._relative_error(scene, chain, protocol="pose_alone")
        self.assertLessEqual(rel, self.TOL, f"initial-pose gradient mismatch: {rel:.2e}")
