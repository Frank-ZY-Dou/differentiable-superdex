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

"""Validation of ``diffsim.set_gravity_backward`` (parameter gradients, part 1).

Ground truth comes from two independent sources, per the project's
anti-cheating discipline:

1. A closed-form derivation for free fall under Backward Euler. With
   ``v_k = v_{k-1} + dt g`` and ``p_k = p_{k-1} + dt v_k``,

       p_N = p_0 + N dt v_0 + dt^2 (N (N + 1) / 2) g,

   so for the loss ``L = 0.5 ||p_N - ref||^2``:

       dL/dg = dt^2 (N (N + 1) / 2) (p_N - ref).

   The engine value must match this formula - any sign or scaling error in
   the C++ implementation fails here and is fixed there, not in the test.
2. Central finite differences of the rollout loss over the gravity vector
   (an entirely separate code path: re-simulate with ``set_gravity(g +/- e)``).

Coverage spans the assembly paths: free rigid (no contact), rigid with
Coulomb contact, articulated pendulum (no contact), and a free-floating
articulated chain on a plane. Also checks the accumulation contract:
``set_gravity_backward`` overwrites (two reads agree) and
``reset_back_propagation`` zeroes the accumulator.

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
    TranslationErrorLoss,
    configure_for_differentiability,
    diffsim,
)

_NUM_WORKER_THREADS = int(os.environ.get("SUPERDEX_DIFFSIM_TEST_THREADS", "0"))
DT = 0.01
NUM_STEPS = 6
FD_EPS = 1e-6


def setUpModule() -> None:
    if not physics.uses_double_precision():
        raise unittest.SkipTest("parameter tests require SUPERDEX_PRECISION=double")
    physics.initialize(num_worker_threads=_NUM_WORKER_THREADS)


def tearDownModule() -> None:
    if physics.is_initialized():
        physics.shutdown()


def adjoint_gravity_gradient(scene, losses) -> np.ndarray:
    """Roll forward, sweep the adjoint, and read dL/d(gravity)."""
    pre, post = [], []
    for _ in range(NUM_STEPS):
        pre.append(scene.capture_state())
        scene.step(DT)
        post.append(scene.capture_state())
    diffsim.reset_back_propagation(scene)
    diffsim.prepare_back_propagate(scene, post[-1], pre[-1])
    for loss in losses:
        loss.accumulate_output_grad()
    for i in range(NUM_STEPS, 0, -1):
        if i != NUM_STEPS:
            diffsim.prepare_back_propagate(scene, post[i - 1], pre[i - 1])
        diffsim.back_propagate(scene)
    grad = np.zeros(3)
    diffsim.set_gravity_backward(scene, grad)
    # Contract: the read is overwriting, so a second read must agree exactly.
    grad_again = np.ones(3)
    diffsim.set_gravity_backward(scene, grad_again)
    np.testing.assert_array_equal(grad, grad_again)
    for handle in pre + post:
        scene.release_state(handle)
    return grad


def fd_gravity_gradient(scene, losses, state_init) -> np.ndarray:
    """Independent ground truth: central differences over set_gravity."""
    gravity0 = np.asarray(scene.get_gravity(), dtype=np.float64)
    fd = np.zeros(3)
    for i in range(3):
        values = []
        for sign in (+1.0, -1.0):
            scene.restore_state(state_init, False)
            g = gravity0.copy()
            g[i] += sign * FD_EPS
            scene.set_gravity(g)
            for _ in range(NUM_STEPS):
                scene.step(DT)
            values.append(sum(loss.value() for loss in losses))
        fd[i] = (values[0] - values[1]) / (2.0 * FD_EPS)
    scene.set_gravity(gravity0)
    return fd


class GravityGradientTest(unittest.TestCase):
    def _check(self, scene, losses, tol: float) -> None:
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        state_init = scene.capture_state()
        grad = adjoint_gravity_gradient(scene, losses)
        fd = fd_gravity_gradient(scene, losses, state_init)
        scene.release_all_states()
        denom = max(np.linalg.norm(grad), np.linalg.norm(fd))
        self.assertGreater(denom, 0.0, "test is vacuous: zero gravity gradient")
        rel = np.linalg.norm(grad - fd) / denom
        self.assertLessEqual(
            rel, tol, f"gravity gradient mismatch: adjoint={grad}, fd={fd}"
        )

    def test_free_fall_closed_form_and_fd(self) -> None:
        """Free rigid body: engine vs the Backward-Euler closed form AND vs FD."""
        scene = physics.create_scene("gravity_free_fall")
        scene.set_gravity([0.0, 0.0, -9.81])
        cube = scene.create_rigid_actor(
            name="cube",
            shape=physics.create_tet_mesh_shape(
                coordinates=scenes.CUBE_COORDS, connectivity=scenes.CUBE_CONN
            ),
            density=1000.0,
            world_from_local=physics.TransformRT([0.0, 0.0, 1.0]),
        )
        cube.set_velocity([0.3, -0.2, 0.1], [0.0, 0.0, 0.0])
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        loss = TranslationErrorLoss(cube)
        state_init = scene.capture_state()

        grad = adjoint_gravity_gradient(scene, [loss])

        # Closed form: dL/dg = dt^2 N(N+1)/2 (p_N - ref), with p_N read from the
        # simulation itself (the scene sits at the final state after the sweep's
        # last prepare restored it; re-simulate cleanly to be explicit).
        scene.restore_state(state_init, False)
        for _ in range(NUM_STEPS):
            scene.step(DT)
        p_final = np.asarray(
            cube.get_center_of_mass_transform().translation, dtype=np.float64
        )
        closed_form = DT * DT * NUM_STEPS * (NUM_STEPS + 1) / 2.0 * (p_final - loss.ref)
        rel_closed = np.linalg.norm(grad - closed_form) / np.linalg.norm(closed_form)
        # Tolerance justification: the engine evaluates dR/dg by central FD with
        # eps_finite_diff = 1e-7; gravity enters the residual linearly, so the
        # remaining error is adjoint-solve precision (abs_tol 1e-10) and FD
        # round-off (~1e-16/1e-7 = 1e-9 relative).
        self.assertLessEqual(
            rel_closed,
            1e-6,
            f"closed-form mismatch: adjoint={grad}, closed_form={closed_form}",
        )

        fd = fd_gravity_gradient(scene, [loss], state_init)
        scene.release_all_states()
        rel_fd = np.linalg.norm(grad - fd) / np.linalg.norm(fd)
        self.assertLessEqual(
            rel_fd, 1e-6, f"FD mismatch: adjoint={grad}, fd={fd}"
        )

    def test_rigid_on_plane_coulomb(self) -> None:
        scene, cube = scenes.rigid_on_plane("coulomb")
        self._check(scene, [TranslationErrorLoss(cube)], 3e-2)

    def test_articulated_pendulum(self) -> None:
        scene, chain = scenes.pendulum(with_controller=False)
        self._check(
            scene, [ArticulatedPoseErrorLoss(chain, np.array([0.4, -0.2]))], 1e-2
        )

    def test_free_chain_on_plane(self) -> None:
        scene, chain = scenes.free_chain_on_plane("coulomb")
        ref = np.zeros(chain.get_num_dofs())
        ref[-1] = 0.3
        self._check(scene, [ArticulatedPoseErrorLoss(chain, ref)], 3e-2)

    def test_reset_zeroes_the_accumulator(self) -> None:
        scene, cube = scenes.rigid_on_plane("coulomb")
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        grad = adjoint_gravity_gradient(scene, [TranslationErrorLoss(cube)])
        self.assertGreater(np.linalg.norm(grad), 0.0)
        diffsim.reset_back_propagation(scene)
        zeroed = np.ones(3)
        diffsim.set_gravity_backward(scene, zeroed)
        np.testing.assert_array_equal(zeroed, np.zeros(3))
        scene.release_all_states()


if __name__ == "__main__":
    unittest.main()
