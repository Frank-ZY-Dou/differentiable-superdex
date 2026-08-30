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

"""Soft-body (FEM) adjoint tests.

Ground truth is central finite differences of the full rollout computed by a
separate forward-only code path, plus two analytic invariants that hold for
any loss on a free-floating body:

- translation invariance: a uniform displacement perturbation passes through
  every Backward-Euler step unchanged (the elastic energy is translation
  invariant and the consistent mass matrix maps the uniform mode to itself in
  the step equations), so the per-axis SUM of dL/du0 equals the per-axis sum
  of the loss gradient, and the per-axis sum of dL/dv0 equals num_steps * dt
  times it. Note the per-NODE closed form dL/du0 = grad only holds for
  mass-weighted loss directions - the elastic Hessian bends everything else -
  which is why these tests validate per-node values against finite
  differences and reserve the closed forms for the invariant sums.
- the free-fall gravity gradient: dL/dg = dt^2 * N(N+1)/2 * sum_nodes(grad)
  per axis, independent of the mass matrix.

Scenes intentionally cover: pure free fall (inertia-only previous-state
coupling), a squashed cube (elastic oscillation, so the previous-state
transport is exercised with active stress), and mass damping (the
GradTarget::Previous mass-damping term). Loud-contract tests pin every 5a
exclusion: soft contact, stiffness damping, missing inertia, Dirichlet BCs,
the rollout-driver gate, and the forced recentering disable.

Requires SUPERDEX_PRECISION=double and a native build with the soft adjoint.
"""

from __future__ import annotations

import os
import unittest

import numpy as np
import superdex.physics as physics

from . import scenes

diffsim = physics.diffsim

_NUM_WORKER_THREADS = int(os.environ.get("SUPERDEX_DIFFSIM_TEST_THREADS", "0"))
DT = 0.01
NUM_STEPS = 3


def setUpModule() -> None:
    if not physics.uses_double_precision():
        raise unittest.SkipTest("soft adjoint tests require SUPERDEX_PRECISION=double")
    physics.initialize(num_worker_threads=_NUM_WORKER_THREADS)


def tearDownModule() -> None:
    if physics.is_initialized():
        physics.shutdown()


def _configure(scene) -> None:
    """Differentiable configuration with a tight forward solve (the adjoint
    assumes the step equations hold exactly; see the harness note)."""
    diffsim.make_scene_differentiable(scene)
    solver = scene.get_solver_params()
    newton = solver.non_linear_solver
    newton.max_iter = 200
    newton.abs_tol = 1e-13
    newton.rel_tol = 1e-13
    solver.non_linear_solver = newton
    scene.set_solver_params(solver)
    params = diffsim.get_back_propagation_solver_params(scene)
    params.validate_finite_diff = True
    params.outer_solver_abs_tol = 1e-12
    params.outer_solver_max_iter = 200
    diffsim.set_back_propagation_solver_params(scene, params)


REF = np.array([0.05, -0.02, 0.11])  # per-axis loss reference offsets


def _loss_grad(cube, num_nodes: int) -> np.ndarray:
    """Gradient of L = 0.5 * sum_nodes |u_node - REF|^2 at the current state."""
    return np.array(cube.get_displacements()) - np.tile(REF, num_nodes)


def _loss_value(cube, num_nodes: int) -> float:
    displacement = np.array(cube.get_displacements()) - np.tile(REF, num_nodes)
    return 0.5 * float(displacement @ displacement)


def _adjoint_sweep(scene, cube, num_steps: int):
    """Forward rollout + reverse sweep with the terminal loss above.

    Returns (grad_u0, grad_v0, grad_gravity, terminal_loss_grad, fd_valid).
    """
    num_nodes = cube.get_num_dofs() // 3
    diffsim.reset_back_propagation(scene)
    pre, post = [], []
    for _ in range(num_steps):
        pre.append(scene.capture_state())
        scene.step(DT)
        post.append(scene.capture_state())
    grad_out = _loss_grad(cube, num_nodes)
    fd_valid = True
    for step in reversed(range(num_steps)):
        diffsim.prepare_back_propagate(scene, post[step], pre[step])
        if step == num_steps - 1:
            diffsim.get_displacements_backward(cube, grad_out)
        diffsim.back_propagate(scene)
        fd_valid = fd_valid and diffsim.get_back_propagation_scene_stats(
            scene
        ).finite_diff_valid
    grad_u0 = np.zeros(3 * num_nodes)
    diffsim.set_displacements_backward(cube, grad_u0)
    grad_v0 = np.zeros(3 * num_nodes)
    diffsim.set_node_velocities_local_backward(cube, grad_v0)
    grad_gravity = np.zeros(3)
    diffsim.set_gravity_backward(scene, grad_gravity)
    for handle in pre + post:
        scene.release_state(handle)
    return grad_u0, grad_v0, grad_gravity, grad_out, fd_valid


def _fd_initial_grads(scene_factory, num_steps: int, eps: float = 1e-6):
    """Central FD of the rollout loss wrt every u0 and v0 component."""

    def rollout_loss(u0, v0) -> float:
        scene, cube = scene_factory()
        try:
            _configure(scene)
            cube.set_displacements(u0)
            cube.set_node_velocities_local(v0)
            for _ in range(num_steps):
                scene.step(DT)
            return _loss_value(cube, cube.get_num_dofs() // 3)
        finally:
            physics.destroy_scene(scene)

    scene, cube = scene_factory()
    u0_base = np.array(cube.get_displacements())
    num_dofs = cube.get_num_dofs()
    physics.destroy_scene(scene)
    # The scene factory always applies the same initial velocities; recover
    # them the same way the factory built them (uniform per axis).
    v0_base = np.tile(np.array([0.3, 0.0, 0.0]), num_dofs // 3)

    fd_u0 = np.zeros(num_dofs)
    fd_v0 = np.zeros(num_dofs)
    for i in range(num_dofs):
        up, um = u0_base.copy(), u0_base.copy()
        up[i] += eps
        um[i] -= eps
        fd_u0[i] = (rollout_loss(up, v0_base) - rollout_loss(um, v0_base)) / (2 * eps)
        vp, vm = v0_base.copy(), v0_base.copy()
        vp[i] += eps
        vm[i] -= eps
        fd_v0[i] = (rollout_loss(u0_base, vp) - rollout_loss(u0_base, vm)) / (2 * eps)
    return fd_u0, fd_v0


class SoftInitialStateGradientTest(unittest.TestCase):
    """Per-node dL/du0 and dL/dv0 against independent rollout FD."""

    def _run_case(self, scene_factory, tol: float) -> None:
        scene, cube = scene_factory()
        self.addCleanup(physics.destroy_scene, scene)
        _configure(scene)
        grad_u0, grad_v0, _, _, fd_valid = _adjoint_sweep(scene, cube, NUM_STEPS)
        self.assertTrue(fd_valid)
        fd_u0, fd_v0 = _fd_initial_grads(scene_factory, NUM_STEPS)
        self.assertGreater(np.abs(fd_u0).max(), 0.0, "test is vacuous")
        self.assertGreater(np.abs(fd_v0).max(), 0.0, "test is vacuous")
        np.testing.assert_allclose(grad_u0, fd_u0, rtol=tol, atol=1e-12)
        np.testing.assert_allclose(grad_v0, fd_v0, rtol=tol, atol=1e-12)

    def test_free_fall_vs_fd(self) -> None:
        # Measured agreement 6e-11 relative (2026-08-30); 1e-7 keeps large
        # headroom above the FD truncation floor.
        self._run_case(lambda: scenes.soft_cube(), 1e-7)

    def test_squashed_cube_vs_fd(self) -> None:
        # Active elastic stress from step one: the previous-state transport
        # interacts with a non-trivial K. FD truncation is larger here.
        self._run_case(lambda: scenes.soft_cube(squash=0.1), 1e-6)

    def test_mass_damping_vs_fd(self) -> None:
        # Exercises the mass-damping term of the GradTarget::Previous
        # assembly (its only other consumer is GradTarget::Current).
        self._run_case(lambda: scenes.soft_cube(mass_damping=2.0, squash=0.1), 1e-6)


class SoftAnalyticInvariantTest(unittest.TestCase):
    """Mass-matrix-free closed forms on the free-fall scene."""

    def test_translation_invariance_and_gravity(self) -> None:
        scene, cube = scenes.soft_cube()
        self.addCleanup(physics.destroy_scene, scene)
        _configure(scene)
        num_nodes = cube.get_num_dofs() // 3
        grad_u0, grad_v0, grad_gravity, grad_out, _ = _adjoint_sweep(
            scene, cube, NUM_STEPS
        )
        sum_grad = grad_out.reshape(-1, 3).sum(axis=0)
        self.assertGreater(np.abs(sum_grad).max(), 0.0, "test is vacuous")

        # Uniform displacement perturbations pass through every step
        # unchanged, so the per-axis sums obey exact closed forms.
        np.testing.assert_allclose(
            grad_u0.reshape(-1, 3).sum(axis=0), sum_grad, rtol=1e-9
        )
        np.testing.assert_allclose(
            grad_v0.reshape(-1, 3).sum(axis=0),
            NUM_STEPS * DT * sum_grad,
            rtol=1e-9,
        )
        # Free-fall gravity gradient (validated for the rigid path in
        # test_diffsim_params; here it also proves the parameter-adjoint
        # residual re-assembly is exact on soft islands).
        closed_gravity = DT * DT * NUM_STEPS * (NUM_STEPS + 1) / 2 * sum_grad
        np.testing.assert_allclose(grad_gravity, closed_gravity, rtol=1e-9)
        self.assertEqual(num_nodes, 8)


class SoftRunningLossTest(unittest.TestCase):
    """Per-step (running) loss accumulation through the soft output backward."""

    def test_running_loss_vs_fd(self) -> None:
        def scene_factory():
            return scenes.soft_cube(squash=0.05)

        def rollout_loss(u0, v0) -> float:
            scene, cube = scene_factory()
            try:
                _configure(scene)
                cube.set_displacements(u0)
                cube.set_node_velocities_local(v0)
                total = 0.0
                for _ in range(NUM_STEPS):
                    scene.step(DT)
                    total += _loss_value(cube, cube.get_num_dofs() // 3)
                return total
            finally:
                physics.destroy_scene(scene)

        scene, cube = scene_factory()
        self.addCleanup(physics.destroy_scene, scene)
        _configure(scene)
        num_nodes = cube.get_num_dofs() // 3
        u0_base = np.array(cube.get_displacements())
        v0_base = np.tile(np.array([0.3, 0.0, 0.0]), num_nodes)

        diffsim.reset_back_propagation(scene)
        pre, post = [], []
        for _ in range(NUM_STEPS):
            pre.append(scene.capture_state())
            scene.step(DT)
            post.append(scene.capture_state())
        for step in reversed(range(NUM_STEPS)):
            diffsim.prepare_back_propagate(scene, post[step], pre[step])
            diffsim.get_displacements_backward(cube, _loss_grad(cube, num_nodes))
            diffsim.back_propagate(scene)
        grad_u0 = np.zeros(3 * num_nodes)
        diffsim.set_displacements_backward(cube, grad_u0)
        for handle in pre + post:
            scene.release_state(handle)

        eps = 1e-6
        fd_u0 = np.zeros(3 * num_nodes)
        for i in range(3 * num_nodes):
            up, um = u0_base.copy(), u0_base.copy()
            up[i] += eps
            um[i] -= eps
            fd_u0[i] = (rollout_loss(up, v0_base) - rollout_loss(um, v0_base)) / (
                2 * eps
            )
        self.assertGreater(np.abs(fd_u0).max(), 0.0, "test is vacuous")
        np.testing.assert_allclose(grad_u0, fd_u0, rtol=1e-6, atol=1e-12)


class SoftContractTest(unittest.TestCase):
    """The 5a exclusions must fail loudly, never degrade silently."""

    def test_recentering_is_force_disabled(self) -> None:
        scene, cube = scenes.soft_cube()
        self.addCleanup(physics.destroy_scene, scene)
        self.assertTrue(cube.get_recentering_params().use_recentering)
        diffsim.make_scene_differentiable(scene)
        self.assertFalse(cube.get_recentering_params().use_recentering)

    def test_stiffness_damping_rejected(self) -> None:
        scene = physics.create_scene("soft_stiffdamp")
        self.addCleanup(physics.destroy_scene, scene)
        scene.set_gravity(scenes.GRAVITY)
        scene.create_soft_actor(
            name="jelly",
            shape=scenes.cube_shape(),
            material=physics.SoftMaterialParams(stiffness_damping_coefficient=0.01),
        )
        with self.assertRaisesRegex(physics.Error, "stiffness damping"):
            diffsim.make_scene_differentiable(scene)

    def test_missing_inertia_rejected(self) -> None:
        scene = physics.create_scene("soft_noinertia")
        self.addCleanup(physics.destroy_scene, scene)
        scene.set_gravity(scenes.GRAVITY)
        scene.create_soft_actor(
            name="jelly",
            shape=scenes.cube_shape(),
            material=physics.SoftMaterialParams(),
            has_inertia=False,
        )
        with self.assertRaisesRegex(physics.Error, "inertia"):
            diffsim.make_scene_differentiable(scene)

    def test_dirichlet_boundary_conditions_rejected(self) -> None:
        scene, cube = scenes.soft_cube()
        self.addCleanup(physics.destroy_scene, scene)
        cube.add_boundary_condition_dofs_world(
            np.array([0, 1, 2], dtype=np.int32), np.zeros(3)
        )
        with self.assertRaisesRegex(physics.Error, "boundary conditions"):
            diffsim.make_scene_differentiable(scene)

    def test_active_soft_contact_rejected_at_back_propagate(self) -> None:
        scene = physics.create_scene("soft_contact")
        self.addCleanup(physics.destroy_scene, scene)
        scene.set_gravity(scenes.GRAVITY)
        scene.create_rigid_actor(
            name="ground",
            shape=physics.create_plane_shape(normal=[0, 0, 1], distance=0.0),
            is_static=True,
            contact=scenes.contact_params("rich"),
        )
        cube = scene.create_soft_actor(
            name="jelly",
            shape=scenes.cube_shape(),
            material=physics.SoftMaterialParams(),
            contact=scenes.contact_params("rich"),
            world_from_local=physics.TransformRT([0.0, 0.0, 0.099]),
        )
        _configure(scene)
        diffsim.reset_back_propagation(scene)
        pre = scene.capture_state()
        scene.step(DT)
        post = scene.capture_state()
        # The cube rests on the plane: contacts are active, so the backward
        # must refuse (its previous-state contact derivative is missing).
        diffsim.prepare_back_propagate(scene, post, pre)
        diffsim.get_displacements_backward(
            cube, _loss_grad(cube, cube.get_num_dofs() // 3)
        )
        with self.assertRaisesRegex(physics.Error, "contact"):
            diffsim.back_propagate(scene)
        scene.release_all_states()

    def test_rollout_driver_gates_soft_actors(self) -> None:
        from superdex.physics.diffsim_rollout import DifferentiableRollout

        scene, _cube = scenes.soft_cube()
        self.addCleanup(physics.destroy_scene, scene)
        _configure(scene)
        with self.assertRaisesRegex(NotImplementedError, "soft actors"):
            DifferentiableRollout(scene, dt=DT, num_steps=2)


if __name__ == "__main__":
    unittest.main()
