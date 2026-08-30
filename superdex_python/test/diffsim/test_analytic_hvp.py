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

- VALID: islands of rigid actors contacting static colliders. The analytic
  operator (exact ``psdDRes = false`` assembly with exact saturation Hessians)
  reproduces the rollout-FD gradients through frictionless, viscous and
  Coulomb contact.
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
    diffsim,
)

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


if __name__ == "__main__":
    unittest.main()
