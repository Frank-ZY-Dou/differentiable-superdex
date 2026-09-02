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

"""Validation of ``diffsim.get_step_jacobian`` (forward mode) against the
reverse-mode adjoint.

``GetStepJacobian`` supports only rigid actors (``mochi_scene.cpp``: "Currently
only rigid actors are supported") and sizes its output by the sum of
``get_num_dofs()`` over every scene actor, so this test uses a single dynamic
cube on a static plane (6 Lie DoFs total).

The adjoint (``back_propagate``) is itself validated against rollout finite
differences in ``test_diffsim_gradients``; here it serves as the reference for
the forward-mode Jacobians. Probing with Lie-basis vectors is exact because
``convert_rigid_gradient_lie_to_quaternion`` -> (internal quaternion-to-Lie in
``get_center_of_mass_transform_backward``) is the identity on the tangent
space (proven by ``GradientConversionRoundTripTest``).

Checked identity, for post-step states s1, s2 of a trajectory started from
the initial state s0: the two-step total
``T = J_curr(2) @ (J_curr(1) + J_old(1)) + J_old(2)`` (the ``J(1)`` terms are
summed because perturbing the initial pose with SetCenterOfMassTransform keeps
the velocity, so the embedded previous configuration moves along) matches the
initial-state gradient read after the full two-sweep adjoint, probed with all
six Lie basis vectors, to a relative tolerance of 1e-2 (the flat matrix
composition ignores the parallel transport between the per-step rotation
charts, and the adjoint's Hessian-vector products are finite-difference
approximations; ~0.4% agreement is observed).

The trajectory starts at rest so that passing the initial state as both
``state_curr`` and ``state_old`` of the first step is exact (zero initial
derived state). Only the end-of-sweep gradient is comparable: the state gradient readable
after a single mid-sweep ``back_propagate`` is NOT the total one-step
transposed-Jacobian product - the velocity-path contribution is held in the
internal ``CDiffDerivedStepGrad`` component and consumed by the next step's
back-propagation (e.g. free translation has dq2/dq1 = 2 for Backward Euler,
while the mid-sweep readout shows the direct term 1).

Requires SUPERDEX_PRECISION=double, like the other diffsim tests.
"""

from __future__ import annotations

import os
import unittest

import numpy as np
import superdex.physics as physics

from . import scenes
from .harness import RIGID_DOF_SIZE, configure_for_differentiability, diffsim

_NUM_WORKER_THREADS = int(os.environ.get("SUPERDEX_DIFFSIM_TEST_THREADS", "0"))
DT = 0.01


def setUpModule() -> None:
    if not physics.uses_double_precision():
        raise unittest.SkipTest("step-Jacobian tests require SUPERDEX_PRECISION=double")
    physics.initialize(num_worker_threads=_NUM_WORKER_THREADS)


def tearDownModule() -> None:
    if physics.is_initialized():
        physics.shutdown()


class StepJacobianTest(unittest.TestCase):
    def _get_step_jacobian(self, scene, s_new, s_curr, s_old, n):
        jac_curr = np.zeros(n * n)
        jac_old = np.zeros(n * n)
        diffsim.get_step_jacobian(scene, s_new, s_curr, s_old, jac_curr, jac_old)
        return (
            jac_curr.reshape((n, n), order="F"),
            jac_old.reshape((n, n), order="F"),
        )

    def _lie_basis_as_output_grad(self, transform, i):
        """Quaternion-representation gradient equivalent to the Lie basis e_i."""
        e_lie = np.zeros(RIGID_DOF_SIZE)
        e_lie[i] = 1.0
        grad7 = np.zeros(7)
        diffsim.convert_rigid_gradient_lie_to_quaternion(transform, e_lie, grad7)
        return grad7

    def _read_state_grad_lie(self, cube, transform):
        """Read the cube's accumulated state gradient, converted to Lie."""
        grad7 = np.zeros(7)
        diffsim.set_center_of_mass_transform_backward(cube, grad7)
        grad_lie = np.zeros(RIGID_DOF_SIZE)
        diffsim.convert_rigid_gradient_quaternion_to_lie(transform, grad7, grad_lie)
        return grad_lie

    def test_rigid_step_jacobian_vs_adjoint(self) -> None:
        # Both steps use the same dt: get_step_jacobian assumes that (see its docs).
        dts = (DT, DT)
        scene, cube = scenes.rigid_on_plane(
            "coulomb", initial_velocity=(0.0, 0.0, 0.0)
        )
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)

        scene_dofs = 0
        actors = []
        scene.for_each_actor(actors.append)
        for actor in actors:
            scene_dofs += actor.get_num_dofs()
        self.assertEqual(scene_dofs, RIGID_DOF_SIZE)  # static plane carries 0
        n = scene_dofs

        # From-rest trajectory: s0 (initial; embedded previous state is itself),
        # then two completed steps.
        s0 = scene.capture_state()
        scene.step(dts[0])
        s1 = scene.capture_state()
        scene.step(dts[1])
        s2 = scene.capture_state()
        transform2 = cube.get_center_of_mass_transform()

        jac2_curr, jac2_old = self._get_step_jacobian(scene, s2, s1, s0, n)
        jac1_curr, jac1_old = self._get_step_jacobian(scene, s1, s0, s0, n)

        # Reference: full two-sweep adjoint probed with each Lie basis vector.
        rows_dq0 = np.zeros((n, n))  # row i = (T^T e_i)^T
        scene.restore_state(s0, False)
        transform0 = cube.get_center_of_mass_transform()
        for i in range(n):
            diffsim.reset_back_propagation(scene)
            diffsim.prepare_back_propagate(scene, s2, s1)
            diffsim.get_center_of_mass_transform_backward(
                cube, self._lie_basis_as_output_grad(transform2, i)
            )
            diffsim.back_propagate(scene)
            diffsim.prepare_back_propagate(scene, s1, s0)
            diffsim.back_propagate(scene)
            rows_dq0[i, :] = self._read_state_grad_lie(cube, transform0)

        scene.release_all_states()

        # Perturbing the initial pose (velocity kept) moves the embedded
        # previous configuration along: T = J2c @ (J1c + J1o) + J2o.
        total = jac2_curr @ (jac1_curr + jac1_old) + jac2_old
        rel = np.linalg.norm(total - rows_dq0) / np.linalg.norm(rows_dq0)
        self.assertLessEqual(
            rel,
            1e-2,
            "two-step chain rule vs adjoint mismatch:\n"
            f"T=\n{total}\nadjoint=\n{rows_dq0}",
        )


if __name__ == "__main__":
    unittest.main()
