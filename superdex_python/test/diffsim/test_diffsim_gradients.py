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

"""End-to-end gradient-consistency tests for ``superdex.physics.diffsim``.

Every test builds a scene from primitives, runs a rollout, computes gradients
w.r.t. the initial pose/velocity, per-step controller targets, and per-step
external forces with the per-step adjoint, and compares each block against a
central finite difference of the whole rollout loss.

Requires the double-precision payload (differentiability is only recommended
in double)::

    cd superdex_python
    SUPERDEX_PRECISION=double python -m unittest test.diffsim.test_diffsim_gradients -v
"""

from __future__ import annotations

import os
import unittest

import numpy as np
import superdex.physics as physics

from .harness import (
    ArticulatedPoseErrorLoss,
    ContactForceLoss,
    GradientCheckCase,
    QuaternionErrorLoss,
    TranslationErrorLoss,
    configure_for_differentiability,
    diffsim,
)
from . import scenes
from superdex.physics.diffsim_rollout import DifferentiableRollout

# Relative-error tolerances, following the internal C++ tests (default 1e-2,
# contact/friction scenes up to 3e-2).
TOL_SMOOTH = 1e-2
TOL_CONTACT = 3e-2


# 0 = single-threaded (default); -1 = all cores, for checking that gradients
# are consistent under the multi-threaded scheduler as well.
_NUM_WORKER_THREADS = int(os.environ.get("SUPERDEX_DIFFSIM_TEST_THREADS", "0"))
_VERBOSE = bool(int(os.environ.get("SUPERDEX_DIFFSIM_VERBOSE", "0")))


def setUpModule() -> None:
    if not physics.uses_double_precision():
        raise unittest.SkipTest(
            "diffsim gradient tests require SUPERDEX_PRECISION=double"
        )
    physics.initialize(num_worker_threads=_NUM_WORKER_THREADS)


def tearDownModule() -> None:
    if physics.is_initialized():
        physics.shutdown()


class DiffsimGradientTest(unittest.TestCase):
    maxDiff = None

    def _check(self, scene, losses, tol, **case_kwargs) -> None:
        self.addCleanup(physics.destroy_scene, scene)
        case = GradientCheckCase(scene, losses, **case_kwargs)
        reports = case.run()
        if _VERBOSE:
            worst = max(reports, key=lambda r: r.rel_error)
            print(
                f"\n[{self._testMethodName}] max_rel_error={worst.rel_error:.3e} "
                f"({worst.name}), adjoint_residual<={case.max_residual:.3e}"
            )
        self.assertTrue(
            case.fd_valid_all,
            "back-propagation finite-difference self-check failed on at least "
            f"one step (max adjoint residual {case.max_residual:.3e})",
        )
        failures = [str(r) for r in reports if r.rel_error > tol]
        self.assertFalse(
            failures,
            "gradient blocks above tolerance "
            f"{tol}:\n" + "\n".join(failures),
        )
        return reports

    # -- sync contact between an articulated link and a dynamic rigid body ----
    #
    # These checks differentiate a 30-step rollout of the chain-pushes-cube scene
    # w.r.t. the pose-controller targets with the rollout driver (no torch), and
    # compare against central finite differences of the rollout loss at two step
    # sizes (1e-5, 1e-6). An entry only counts when the two FD estimates agree
    # to 1e-4 (at the contact onset the loss can be rougher than that), so that
    # a rough loss can never hide a wrong adjoint nor fake one; at least two
    # entries must count.

    _ONSET_ENTRIES = (-1, 0, 1, 5)  # steps relative to the cube's first motion

    def _chain_control_gradient_check(self, scene, chain, cube, speed=-1.2, num_steps=30):
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        dt = 0.01
        goal = np.array([0.5, 0.0, 0.1])
        controls = np.stack(
            [np.linspace(0.0, speed, num_steps), np.zeros(num_steps)], axis=1
        )

        def loss_value():
            d = np.asarray(cube.get_center_of_mass_transform().translation) - goal
            return 0.5 * float(d @ d)

        class CubeLoss:
            def value(self):
                return loss_value()

            def accumulate_output_grad(self):
                d = np.asarray(cube.get_center_of_mass_transform().translation) - goal
                g = np.zeros(7)
                g[:3] = d
                diffsim.get_center_of_mass_transform_backward(cube, g)

        state0 = scene.capture_state()

        def rollout_loss(c):
            scene.restore_state(state0, False)
            for step in range(num_steps):
                chain.set_articulated_target_pose(np.ascontiguousarray(c[step]))
                scene.step(dt)
            return loss_value()

        # Forward once to find the step at which the cube starts moving.
        scene.restore_state(state0, False)
        onset = None
        for step in range(num_steps):
            chain.set_articulated_target_pose(np.ascontiguousarray(controls[step]))
            scene.step(dt)
            x = np.asarray(cube.get_center_of_mass_transform().translation)[0]
            if onset is None and abs(x - 0.25) > 1e-6:
                onset = step
        self.assertIsNotNone(onset, "test is vacuous: the chain never moved the cube")

        scene.restore_state(state0, False)
        rollout = DifferentiableRollout(scene, dt=dt, num_steps=num_steps)
        result = rollout.run(
            apply_inputs=lambda step: chain.set_articulated_target_pose(
                np.ascontiguousarray(controls[step])
            ),
            terminal_losses=[CubeLoss()],
        )
        adjoint = result.gradients["chain"].control_targets  # (2, num_steps)

        rel_errors = {}
        skipped = {}
        for offset in self._ONSET_ENTRIES:
            step = onset + offset
            fds = []
            for eps in (1e-5, 1e-6):
                cp = controls.copy()
                cp[step, 0] += eps
                cm = controls.copy()
                cm[step, 0] -= eps
                fds.append((rollout_loss(cp) - rollout_loss(cm)) / (2.0 * eps))
            scale = max(abs(fds[0]), 1e-12)
            fd_self = abs(fds[0] - fds[1]) / scale
            if fd_self > 1e-4:
                skipped[step] = fd_self
                continue
            rel_errors[step] = abs(adjoint[0, step] - fds[0]) / scale
        scene.release_state(state0)
        self.assertGreaterEqual(
            len(rel_errors),
            2,
            f"too few smooth entries to judge the adjoint (skipped: {skipped})",
        )
        return rel_errors

    def test_frictional_contact_with_rotating_surfaces_needs_fade_friction_off(self):
        """The alignment fading of friction (``fade_friction``) scales the friction
        normal force by a factor that depends on the stage-start orientations of
        both bodies; the previous-state adjoint assembly treats that factor as a
        constant. Differentiable scenes therefore override fade_friction to
        false (mochi_scene.cpp), and with the override the adjoint of a chain
        pushing a tipping cube (cube-ground friction, frictionless chain) is
        exact. Measured 2026-09-01 on this scene: with fading 3e-4..1.8e-2
        relative error at the contact onset, without 1e-7..2e-6."""
        frictionless = physics.ContactParams(
            penalty_coefficient=1e8, coulomb_friction_coefficient=0.0
        )
        scene, chain, cube = scenes.chain_pushing_cube_with_params(
            scenes.contact_params("rich"), chain_cp=frictionless
        )
        self.addCleanup(physics.destroy_scene, scene)
        sp = scene.get_solver_params()
        ee = sp.experimental_eval
        ee.fade_friction = True
        sp.experimental_eval = ee
        scene.set_solver_params(sp)
        diffsim.make_scene_differentiable(scene)
        self.assertFalse(
            scene.get_solver_params().experimental_eval.fade_friction,
            "make_scene_differentiable must switch fade_friction off",
        )
        sp = scene.get_solver_params()
        ee = sp.experimental_eval
        ee.fade_friction = True
        sp.experimental_eval = ee
        scene.set_solver_params(sp)
        self.assertFalse(
            scene.get_solver_params().experimental_eval.fade_friction,
            "set_solver_params must keep fade_friction off in a differentiable scene",
        )
        rel_errors = self._chain_control_gradient_check(scene, chain, cube)
        self.assertLessEqual(max(rel_errors.values()), 1e-4, rel_errors)

    def test_frictional_contact_through_sdf_edge_regions_is_exact(self):
        """With explicit normals the dissipative contact terms use the collider's
        stage-start SDF gradient as the contact normal. Until 2026-09-01 the
        GradTarget::Previous assembly treated that normal as a constant, which
        is exact inside a face region of a box SDF and wrong where contact
        samples slide through an edge or corner region (gradient varying with
        position): about 1e-3 relative per step, compounding over a rollout
        (measured here: 1e-2..2e-1 with the cube's face samples on the tilted
        link's edge region, 3e-7 with the link's edge samples on the cube's
        face). The previous-state assembly now differentiates the normal with
        the stage-start SDF Hessian (contact_utils.h,
        ComputeBatchContactDissipationForceDForce; Hessians from grid SDFs and
        the plane/sphere/box primitives), and both directions are exact:
        measured 1e-6..4e-5 (the largest at an entry whose FD self-consistency
        is 1e-4) and 3e-7."""
        viscous = physics.ContactParams(
            penalty_coefficient=1e8,
            coulomb_friction_coefficient=0.0,
            viscous_friction_coefficient=0.1,
        )
        for label, kwargs in (
            ("link edge samples on the cube's face region", {"link_collider": False}),
            ("cube face samples on the tilted link's edge region", {"cube_collider": False}),
            ("both bodies collide", {}),
        ):
            with self.subTest(label):
                scene, chain, cube = scenes.chain_pushing_cube_with_params(viscous, **kwargs)
                rel_errors = self._chain_control_gradient_check(scene, chain, cube)
                self.assertLessEqual(max(rel_errors.values()), 1e-4, rel_errors)

    # -- rigid ---------------------------------------------------------------

    def test_rigid_free_translation(self) -> None:
        scene, cube = scenes.rigid_free()
        self._check(scene, [TranslationErrorLoss(cube)], TOL_SMOOTH)

    def test_rigid_free_rotation(self) -> None:
        scene, cube = scenes.rigid_free()
        self._check(scene, [QuaternionErrorLoss(cube)], TOL_SMOOTH)

    def test_rigid_on_plane_frictionless(self) -> None:
        scene, cube = scenes.rigid_on_plane("none")
        self._check(scene, [TranslationErrorLoss(cube)], TOL_CONTACT)

    def test_rigid_on_plane_viscous_friction(self) -> None:
        scene, cube = scenes.rigid_on_plane("viscous")
        self._check(scene, [TranslationErrorLoss(cube)], TOL_CONTACT)

    def test_rigid_on_plane_coulomb_friction(self) -> None:
        scene, cube = scenes.rigid_on_plane("coulomb")
        self._check(scene, [TranslationErrorLoss(cube)], TOL_CONTACT)

    def test_two_cubes_on_plane_coulomb(self) -> None:
        scene, bottom = scenes.two_cubes_on_plane("coulomb")
        self._check(scene, [TranslationErrorLoss(bottom)], TOL_CONTACT)

    def test_chain_pushing_cube_sync_contact(self) -> None:
        """Articulated link vs dynamic rigid cube in one island (sync contact),
        with the loss on the pushed cube: gradients w.r.t. the chain's initial
        state and per-step controller targets. Measured 2026-09-01 on the
        torch-bridge probe: 1.4e-5..2.6e-3 relative against central FD that was
        self-consistent to 2e-6 - the same class as the other contact scenes."""
        scene, chain, cube = scenes.chain_pushing_cube("rich")
        reports = self._check(
            scene,
            [TranslationErrorLoss(cube, ref=np.array([0.5, 0.0, 0.1]))],
            TOL_CONTACT,
            num_steps=30,
            dt=0.01,
            control_speed=np.array([-1.5, 0.0]),  # swing the root joint only
            force_speed=0.0,
        )
        # Vacuousness guard: the loss lives on the cube, so a nonzero control
        # gradient proves the chain actually pushed it during the rollout.
        control_reports = [r for r in reports if r.name.startswith("control")]
        self.assertTrue(control_reports)
        self.assertGreater(
            max(float(np.abs(r.finite_diff).max()) for r in control_reports),
            1e-6,
            "test is vacuous: the chain never moved the cube",
        )

    # -- articulated ---------------------------------------------------------

    def test_pendulum_free_motion(self) -> None:
        scene, chain = scenes.pendulum(with_controller=False)
        ref = np.array([0.4, -0.2])
        self._check(scene, [ArticulatedPoseErrorLoss(chain, ref)], TOL_SMOOTH)

    def test_pendulum_with_controller(self) -> None:
        scene, chain = scenes.pendulum(with_controller=True)
        ref = np.array([0.4, -0.2])
        self._check(
            scene,
            [ArticulatedPoseErrorLoss(chain, ref)],
            TOL_SMOOTH,
            control_speed=0.5,
        )

    def test_free_chain_on_plane_coulomb(self) -> None:
        scene, chain = scenes.free_chain_on_plane("coulomb")
        ref = np.zeros(chain.get_num_dofs())
        ref[-1] = 0.3
        self._check(scene, [ArticulatedPoseErrorLoss(chain, ref)], TOL_CONTACT)

    # -- contact-force output backward ---------------------------------------

    def test_contact_force_loss(self) -> None:
        """Loss on the total world contact force (get_contact_force_world_backward)."""
        scene, cube = scenes.rigid_on_plane("coulomb")
        self._check(scene, [ContactForceLoss(cube)], TOL_CONTACT)


class GradientConversionRoundTripTest(unittest.TestCase):
    """convert_rigid_gradient_quaternion_to_lie is the transpose-chain partner
    of convert_rigid_gradient_lie_to_quaternion; for any Lie-tangent gradient,
    lie -> quaternion -> lie must be the identity (the quaternion detour only
    adds/removes the radial null direction)."""

    def test_lie_quaternion_round_trip(self) -> None:
        rng = np.random.default_rng(1234)
        for _ in range(10):
            rotation = physics.Quaternion.from_rotation_vector(
                rng.uniform(-1.5, 1.5, size=3)
            )
            transform = physics.TransformRT(
                translation=rng.uniform(-1.0, 1.0, size=3), rotation=rotation
            )
            grad_lie = rng.standard_normal(6)
            grad_quat = np.zeros(7)
            diffsim.convert_rigid_gradient_lie_to_quaternion(
                transform, grad_lie, grad_quat
            )
            grad_lie_back = np.zeros(6)
            diffsim.convert_rigid_gradient_quaternion_to_lie(
                transform, grad_quat, grad_lie_back
            )
            np.testing.assert_allclose(
                grad_lie_back, grad_lie, rtol=1e-10, atol=1e-12
            )


class TargetVelocityBackwardTest(unittest.TestCase):
    """set_articulated_target_velocity_backward vs finite differences.

    The target velocity is applied once before the first step (it initializes
    the previous controller target as target_old = target_current - dt * v),
    so its gradient is read after the back_propagate of step 0, i.e. at the
    end of the reverse sweep.
    """

    NUM_STEPS = 6
    DT = 0.01
    FD_EPS = 1e-6

    def _rollout(self, scene, chain, pose0, target_velocity):
        chain.set_articulated_target_velocity(np.asarray(target_velocity))
        pre, post = [], []
        for _ in range(self.NUM_STEPS):
            chain.set_articulated_target_pose(pose0)
            pre.append(scene.capture_state())
            scene.step(self.DT)
            post.append(scene.capture_state())
        return pre, post

    def _loss_and_pose(self, chain, ref):
        pose = np.zeros(chain.get_num_dofs())
        chain.get_articulated_pose(pose)
        diff = pose - ref
        return 0.5 * float(diff @ diff), diff

    def test_target_velocity_gradient(self) -> None:
        scene, chain = scenes.pendulum(with_controller=True)
        self.addCleanup(physics.destroy_scene, scene)
        from .harness import configure_for_differentiability

        configure_for_differentiability(scene)
        num_dofs = chain.get_num_dofs()
        pose0 = np.zeros(num_dofs)
        chain.get_articulated_pose(pose0)
        ref = np.array([0.4, -0.2])
        state_init = scene.capture_state()
        target_velocity = np.array([0.3, -0.1])

        # Adjoint.
        pre, post = self._rollout(scene, chain, pose0, target_velocity)
        _, diff = self._loss_and_pose(chain, ref)
        diffsim.reset_back_propagation(scene)
        diffsim.prepare_back_propagate(scene, post[-1], pre[-1])
        diffsim.get_articulated_pose_backward(chain, diff)
        for i in range(self.NUM_STEPS, 0, -1):
            if i != self.NUM_STEPS:
                diffsim.prepare_back_propagate(scene, post[i - 1], pre[i - 1])
            diffsim.back_propagate(scene)
        grad_velocity = np.zeros(num_dofs)
        diffsim.set_articulated_target_velocity_backward(chain, grad_velocity)

        # Central finite differences of the rollout loss.
        fd = np.zeros(num_dofs)
        for j in range(num_dofs):
            values = []
            for sign in (+1.0, -1.0):
                scene.restore_state(state_init, False)
                perturbed = target_velocity.copy()
                perturbed[j] += sign * self.FD_EPS
                self._rollout(scene, chain, pose0, perturbed)
                values.append(self._loss_and_pose(chain, ref)[0])
            fd[j] = (values[0] - values[1]) / (2.0 * self.FD_EPS)
        scene.release_all_states()

        denom = max(np.linalg.norm(grad_velocity), np.linalg.norm(fd))
        self.assertGreater(denom, 0.0)
        rel = np.linalg.norm(grad_velocity - fd) / denom
        self.assertLessEqual(
            rel,
            TOL_SMOOTH,
            f"target-velocity gradient mismatch: analytic={grad_velocity}, fd={fd}",
        )


if __name__ == "__main__":
    unittest.main()
