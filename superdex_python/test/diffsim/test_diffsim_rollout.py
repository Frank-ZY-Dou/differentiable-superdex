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
  finite differences, and truncation withholds the initial-state gradients.

Requires SUPERDEX_PRECISION=double.
"""

from __future__ import annotations

import os
import unittest

import numpy as np
import superdex.physics as physics
from superdex.physics.diffsim_rollout import DifferentiableRollout

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


def setUpModule() -> None:
    if not physics.uses_double_precision():
        raise unittest.SkipTest("rollout tests require SUPERDEX_PRECISION=double")
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
    """Per-step losses validated against finite differences of the summed cost."""

    FD_EPS = 1e-6

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
            1e-4,
            f"running-loss gradient mismatch: analytic={grads.initial_velocity}, fd={fd}",
        )


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
