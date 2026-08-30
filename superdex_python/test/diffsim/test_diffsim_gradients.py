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

import unittest

import numpy as np
import superdex.physics as physics

from .harness import (
    ArticulatedPoseErrorLoss,
    GradientCheckCase,
    QuaternionErrorLoss,
    TranslationErrorLoss,
)
from . import scenes

# Relative-error tolerances, following the internal C++ tests (default 1e-2,
# contact/friction scenes up to 3e-2).
TOL_SMOOTH = 1e-2
TOL_CONTACT = 3e-2


def setUpModule() -> None:
    if not physics.uses_double_precision():
        raise unittest.SkipTest(
            "diffsim gradient tests require SUPERDEX_PRECISION=double"
        )
    physics.initialize(num_worker_threads=0)


def tearDownModule() -> None:
    if physics.is_initialized():
        physics.shutdown()


class DiffsimGradientTest(unittest.TestCase):
    maxDiff = None

    def _check(self, scene, losses, tol, **case_kwargs) -> None:
        self.addCleanup(physics.destroy_scene, scene)
        case = GradientCheckCase(scene, losses, **case_kwargs)
        reports = case.run()
        self.assertTrue(
            case.stats.finite_diff_valid,
            "back-propagation finite-difference self-check failed "
            f"(residual={case.stats.residual_norm:.3e})",
        )
        failures = [str(r) for r in reports if r.rel_error > tol]
        self.assertFalse(
            failures,
            "gradient blocks above tolerance "
            f"{tol}:\n" + "\n".join(failures),
        )

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


if __name__ == "__main__":
    unittest.main()
