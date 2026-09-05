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

"""Validity boundary of the experimental analytic Hessian-vector-product
operator (``BackPropagationSolverParams.use_analytic_hvp``).

Pins down what was measured when the flag was added:

- VALID at the harness's 3e-2 tolerance: islands of rigid actors contacting
  static colliders. The analytic operator (``psdDRes = false`` assembly with
  exact saturation Hessians) reproduces the rollout-FD gradients through
  frictionless, viscous and Coulomb contact at that level only: measured
  with the two-step-size rollout protocol (AnalyticHvpPrecisionPinTest,
  2026-09-01) it is off by 7e-4 (rich friction) to 4e-2 (frictionless)
  relative on the sliding cube, where the finite-difference operator is
  exact to 1e-7. The assembled matrix is exactly symmetric, so the missing
  part is symmetric too (Gauss-Newton-grade rotational coupling).
- INVALID by design of the current assembly: articulated actors (the
  assembled dresidual is not the true residual derivative there - the same
  reason ``get_step_jacobian`` is rigid-only) and dynamic-dynamic contact
  coupling. The divergence from rollout finite differences is asserted here,
  so a future assembly fix flips this test rather than silently changing
  behavior. (The built-in single-probe cross-check is a sampling diagnostic:
  it caught the fitted-saturation contact mismatch, but a probe along the
  solution direction can miss the articulated operator error, so detection is
  asserted on the gradients themselves.)

Skips (rather than failing) on builds whose extension predates the flag.
Requires SUPERDEX_PRECISION=double.
"""

from __future__ import annotations

import os
import unittest

import numpy as np
import superdex.physics as physics

from . import scenes
from .harness import (
    ArticulatedPoseErrorLoss,
    GradientCheckCase,
    TranslationErrorLoss,
    configure_for_differentiability,
    diffsim,
)
from superdex.physics.diffsim_rollout import DifferentiableRollout

_NUM_WORKER_THREADS = int(os.environ.get("SUPERDEX_DIFFSIM_TEST_THREADS", "0"))
TOL_CONTACT = 3e-2


def setUpModule() -> None:
    if not physics.uses_double_precision():
        raise unittest.SkipTest("analytic-HVP tests require SUPERDEX_PRECISION=double")
    if not hasattr(diffsim.BackPropagationSolverParams(), "use_analytic_hvp"):
        raise unittest.SkipTest("this build predates use_analytic_hvp")
    physics.initialize(num_worker_threads=_NUM_WORKER_THREADS)


def tearDownModule() -> None:
    if physics.is_initialized():
        physics.shutdown()


def _enable_analytic(scene) -> None:
    dp = diffsim.get_back_propagation_solver_params(scene)
    dp.use_analytic_hvp = True
    dp.validate_finite_diff = True
    diffsim.set_back_propagation_solver_params(scene, dp)


class AnalyticHvpValidDomainTest(unittest.TestCase):
    """Rigid actor vs static collider: analytic operator must match rollout FD."""

    def _check(self, scene, losses) -> None:
        self.addCleanup(physics.destroy_scene, scene)
        case = GradientCheckCase(scene, losses)
        _enable_analytic(scene)
        reports = case.run()
        self.assertTrue(
            case.fd_valid_all,
            "analytic-vs-FD cross-check fired inside the valid domain "
            f"(max adjoint residual {case.max_residual:.3e})",
        )
        failures = [str(r) for r in reports if r.rel_error > TOL_CONTACT]
        self.assertFalse(
            failures, "analytic-mode gradients off in the valid domain:\n" + "\n".join(failures)
        )

    def test_rigid_on_plane_frictionless(self) -> None:
        scene, cube = scenes.rigid_on_plane("none")
        self._check(scene, [TranslationErrorLoss(cube)])

    def test_rigid_on_plane_viscous(self) -> None:
        scene, cube = scenes.rigid_on_plane("viscous")
        self._check(scene, [TranslationErrorLoss(cube)])

    def test_rigid_on_plane_coulomb(self) -> None:
        scene, cube = scenes.rigid_on_plane("coulomb")
        self._check(scene, [TranslationErrorLoss(cube)])


class AnalyticHvpLimitDetectionTest(unittest.TestCase):
    """Outside the valid domain, analytic-mode gradients must visibly diverge.

    If a future change makes the articulated assembly exact, this test starts
    failing - then move the articulated scenes into the valid-domain test and
    update the flag's documentation.
    """

    def test_articulated_mismatch_is_detected(self) -> None:
        scene, chain = scenes.pendulum(with_controller=False)
        self.addCleanup(physics.destroy_scene, scene)
        case = GradientCheckCase(
            scene, [ArticulatedPoseErrorLoss(chain, np.array([0.4, -0.2]))]
        )
        _enable_analytic(scene)
        reports = case.run()
        worst = max(r.rel_error for r in reports)
        self.assertGreater(
            worst,
            0.05,
            "articulated analytic-mode gradients unexpectedly match rollout "
            "finite differences; if the assembly became exact, promote "
            "articulated scenes to the valid domain and update the "
            "use_analytic_hvp documentation",
        )


class AnalyticHvpPrecisionPinTest(unittest.TestCase):
    """The accuracies behind the valid-domain claim, measured with rollout
    central finite differences of the loss w.r.t. the cube's initial velocity
    at two step sizes (1e-5, 1e-6; each component must be self-consistent to
    1e-5). Sliding cube on a static plane, 2026-09-01: the finite-difference
    operator's gradient is within 3e-8 (rich friction) / 2e-7 (frictionless)
    of the rollout gradient in norm; the analytic operator's is off by 7e-4 /
    3.5e-2. A fix of the assembly flips the second assertion - then promote
    the operator in the documentation instead of loosening anything here."""

    NUM_STEPS = 30
    DT = 0.01
    V0 = np.array([0.5, 0.0, 0.0])
    GOAL = np.array([0.5, 0.0, 0.1])

    class CubeLoss:
        def __init__(self, cube, goal):
            self.cube, self.goal = cube, goal

        def value(self):
            d = np.asarray(self.cube.get_center_of_mass_transform().translation) - self.goal
            return 0.5 * float(d @ d)

        def accumulate_output_grad(self):
            d = np.asarray(self.cube.get_center_of_mass_transform().translation) - self.goal
            g = np.zeros(7)
            g[:3] = d
            diffsim.get_center_of_mass_transform_backward(self.cube, g)

    def _adjoint(self, friction: str, analytic: bool) -> np.ndarray:
        scene, cube = scenes.rigid_on_plane(friction, initial_velocity=tuple(self.V0))
        configure_for_differentiability(scene)
        dp = diffsim.get_back_propagation_solver_params(scene)
        dp.validate_finite_diff = True
        dp.use_analytic_hvp = analytic
        diffsim.set_back_propagation_solver_params(scene, dp)
        result = DifferentiableRollout(scene, dt=self.DT, num_steps=self.NUM_STEPS).run(
            apply_inputs=lambda step: None, terminal_losses=[self.CubeLoss(cube, self.GOAL)]
        )
        g = result.gradients["cube"].initial_velocity[:3].copy()
        physics.destroy_scene(scene)
        return g

    def _rollout_loss(self, friction: str, v0: np.ndarray) -> float:
        scene, cube = scenes.rigid_on_plane(friction, initial_velocity=tuple(v0))
        configure_for_differentiability(scene)
        loss = self.CubeLoss(cube, self.GOAL)
        for _ in range(self.NUM_STEPS):
            scene.step(self.DT)
        value = loss.value()
        physics.destroy_scene(scene)
        return value

    def _rollout_fd(self, friction: str) -> np.ndarray:
        fd = np.zeros(3)
        for k in range(3):
            estimates = []
            for h in (1e-5, 1e-6):
                dv = np.zeros(3)
                dv[k] = h
                estimates.append(
                    (self._rollout_loss(friction, self.V0 + dv) - self._rollout_loss(friction, self.V0 - dv))
                    / (2.0 * h)
                )
            self.assertLessEqual(
                abs(estimates[0] - estimates[1]) / abs(estimates[0]),
                1e-5,
                f"rollout FD not self-consistent for dv0[{k}] ({friction})",
            )
            fd[k] = estimates[0]
        return fd

    def test_fd_operator_is_exact_and_the_analytic_operator_is_not(self) -> None:
        for friction in ("rich", "none"):
            with self.subTest(friction):
                fd = self._rollout_fd(friction)
                rel_fd = np.linalg.norm(self._adjoint(friction, False) - fd) / np.linalg.norm(fd)
                rel_analytic = np.linalg.norm(self._adjoint(friction, True) - fd) / np.linalg.norm(fd)
                self.assertLessEqual(rel_fd, 1e-6, f"finite-difference operator: {rel_fd:.2e}")
                self.assertGreaterEqual(
                    rel_analytic, 1e-4, f"analytic operator unexpectedly exact: {rel_analytic:.2e}"
                )


if __name__ == "__main__":
    unittest.main()
