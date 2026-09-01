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
GradTarget::Previous mass-damping term).

Contact (phase 5b): a soft cube sliding on a static plane exercises the
async-contact adjoint - the GradTarget::Previous contact assembly on the soft
side (d merit / d u_prev through the friction/damping terms) plus collision
detection inside the finite-difference Hessian-vector products. Every
friction regime, the sticking regime, a long rollout and a cube that lands
mid-rollout are checked against rollout finite differences; so are the
contact-parameter and gravity gradients on the contact scene. One test pins
the approximation the adjoint shares with the rigid path: the derivative of
the stage-start colliding normal (explicit normals) with respect to the
previous state is dropped, which is exact for flat contact and O(sin(tilt) /
(max_alignment_normals + 1)) otherwise - a probe with amplified alignment
fading on a tilted cube must show that error (upper AND lower bound).

Loud-contract tests pin the remaining exclusions: sync (dynamic-dynamic)
contact involving a soft actor, stiffness damping, missing inertia, Dirichlet
BCs, the rollout-driver gate, and the forced recentering disable.

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
V0 = (0.3, 0.0, 0.0)  # default initial nodal velocity of the scene factories


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


def _fd_initial_grads(
    scene_factory, num_steps: int, eps: float = 1e-6, initial_velocity=V0
):
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
    # The scene factories apply uniform per-axis initial velocities; rebuild
    # the same vector here (the state read-back is not needed for that).
    v0_base = np.tile(np.asarray(initial_velocity, dtype=np.float64), num_dofs // 3)

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


class _InitialStateFdMixin:
    """Per-node dL/du0 and dL/dv0 against independent rollout FD."""

    def _run_case(
        self, scene_factory, tol: float, num_steps: int = NUM_STEPS, initial_velocity=V0
    ) -> None:
        scene, cube = scene_factory()
        self.addCleanup(physics.destroy_scene, scene)
        _configure(scene)
        grad_u0, grad_v0, _, _, fd_valid = _adjoint_sweep(scene, cube, num_steps)
        self.assertTrue(fd_valid)
        fd_u0, fd_v0 = _fd_initial_grads(
            scene_factory, num_steps, initial_velocity=initial_velocity
        )
        self.assertGreater(np.abs(fd_u0).max(), 0.0, "test is vacuous")
        self.assertGreater(np.abs(fd_v0).max(), 0.0, "test is vacuous")
        np.testing.assert_allclose(grad_u0, fd_u0, rtol=tol, atol=1e-12)
        np.testing.assert_allclose(grad_v0, fd_v0, rtol=tol, atol=1e-12)


class SoftInitialStateGradientTest(_InitialStateFdMixin, unittest.TestCase):
    """Free-floating cubes: inertia, elastic stress and mass-damping transport."""

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

    def _soft_on_plane_with_rigid(self, rigid_position):
        """Soft cube on the plane plus a dynamic rigid cube at `rigid_position`."""
        scene, cube = scenes.soft_cube_on_plane("rich")
        scene.create_rigid_actor(
            name="rigid",
            shape=scenes.cube_shape(),
            density=1000.0,
            contact=scenes.contact_params("rich"),
            world_from_local=physics.TransformRT(list(rigid_position)),
        )
        return scene, cube

    def _one_step_backward(self, scene, cube) -> np.ndarray:
        _configure(scene)
        diffsim.reset_back_propagation(scene)
        pre = scene.capture_state()
        scene.step(DT)
        post = scene.capture_state()
        diffsim.prepare_back_propagate(scene, post, pre)
        diffsim.get_displacements_backward(
            cube, _loss_grad(cube, cube.get_num_dofs() // 3)
        )
        diffsim.back_propagate(scene)
        grad_u0 = np.zeros(cube.get_num_dofs())
        diffsim.set_displacements_backward(cube, grad_u0)
        scene.release_all_states()
        return grad_u0

    def test_sync_contact_with_dynamic_actor_rejected(self) -> None:
        # A rigid cube resting on the soft cube: both share an island (sync
        # contact), whose soft-side previous-state derivative is missing, so
        # the backward must refuse instead of dropping the term.
        scene, cube = self._soft_on_plane_with_rigid((0.0, 0.0, 0.298))
        self.addCleanup(physics.destroy_scene, scene)
        with self.assertRaisesRegex(physics.Error, "sync contact"):
            self._one_step_backward(scene, cube)

    def test_dynamic_actor_in_separate_island_is_allowed(self) -> None:
        # The same rigid cube far away lives in its own island: the soft
        # cube's contact against the static plane is async contact, which is
        # supported, and the bystander must not change the soft gradient.
        scene_alone, cube_alone = scenes.soft_cube_on_plane("rich")
        self.addCleanup(physics.destroy_scene, scene_alone)
        grad_alone = self._one_step_backward(scene_alone, cube_alone)
        scene, cube = self._soft_on_plane_with_rigid((3.0, 0.0, 0.099))
        self.addCleanup(physics.destroy_scene, scene)
        grad = self._one_step_backward(scene, cube)
        self.assertGreater(np.abs(grad_alone).max(), 0.0, "test is vacuous")
        np.testing.assert_allclose(grad, grad_alone, rtol=1e-12, atol=0.0)

    def test_rollout_driver_gates_soft_actors(self) -> None:
        from superdex.physics.diffsim_rollout import DifferentiableRollout

        scene, _cube = scenes.soft_cube()
        self.addCleanup(physics.destroy_scene, scene)
        _configure(scene)
        with self.assertRaisesRegex(NotImplementedError, "soft actors"):
            DifferentiableRollout(scene, dt=DT, num_steps=2)


def _final_displacements(scene_factory, num_steps: int = NUM_STEPS) -> np.ndarray:
    scene, cube = scene_factory()
    try:
        _configure(scene)
        for _ in range(num_steps):
            scene.step(DT)
        return np.array(cube.get_displacements())
    finally:
        physics.destroy_scene(scene)


class SoftContactGradientTest(_InitialStateFdMixin, unittest.TestCase):
    """dL/du0 and dL/dv0 of a soft cube sliding on a static plane, vs rollout FD.

    Static colliders are async contact in the engine: this is the regime the
    soft contact adjoint implements (GradTarget::Previous contact assembly on
    the soft side plus collision detection inside the FD Hessian-vector
    products). Measured agreement on 2026-09-01: 1.3e-6..5.3e-6 relative
    across the four regimes at 3 steps, 1.2e-5 at 8 steps, 5.8e-7 in the
    sticking regime, 2.4e-6 for a cube landing mid-rollout. The tolerance of
    1e-4 keeps >= 8x headroom above the worst measurement; the FD truncation
    floor at eps = 1e-6 is ~1e-6 relative.
    """

    TOL = 1e-4

    def test_contact_is_active_and_friction_matters(self) -> None:
        """Vacuousness guards for the whole class, checked once explicitly."""
        on_plane = _final_displacements(lambda: scenes.soft_cube_on_plane("rich"))
        free = _final_displacements(
            lambda: scenes.soft_cube(initial_velocity=V0)
        )
        # Contact must change the trajectory (free fall would sink ~6 mm).
        self.assertGreater(np.abs(on_plane - free).max(), 1e-3)
        # Friction must decelerate the slide: the bottom nodes travel less in
        # x than the frictionless kinematic prediction v0 * N * dt.
        rest_z = scenes.CUBE_COORDS.reshape(-1, 3)[:, 2]
        bottom = rest_z < 0.0
        travel_x = on_plane.reshape(-1, 3)[bottom, 0].mean()
        self.assertLess(travel_x, 0.95 * V0[0] * NUM_STEPS * DT)
        self.assertGreater(travel_x, 0.0)

    def test_penalty_only_regime_vs_fd(self) -> None:
        self._run_case(lambda: scenes.soft_cube_on_plane("none"), self.TOL)

    def test_viscous_friction_vs_fd(self) -> None:
        self._run_case(lambda: scenes.soft_cube_on_plane("viscous"), self.TOL)

    def test_coulomb_friction_vs_fd(self) -> None:
        self._run_case(lambda: scenes.soft_cube_on_plane("coulomb"), self.TOL)

    def test_all_dissipation_terms_vs_fd(self) -> None:
        self._run_case(lambda: scenes.soft_cube_on_plane("rich"), self.TOL)

    def test_sticking_regime_vs_fd(self) -> None:
        # Tangential motion per step far below friction_falloff_vel * dt: the
        # regularized Coulomb term is in its quadratic (sticking) branch.
        v0 = (0.002, 0.001, 0.0)
        self._run_case(
            lambda: scenes.soft_cube_on_plane("rich", initial_velocity=v0),
            self.TOL,
            initial_velocity=v0,
        )

    def test_long_rollout_vs_fd(self) -> None:
        self._run_case(lambda: scenes.soft_cube_on_plane("rich"), self.TOL, num_steps=8)

    def test_landing_mid_rollout_vs_fd(self) -> None:
        # Starts 6 mm above the plane and lands after a few steps, so the
        # contact set changes inside the differentiated window.
        self._run_case(
            lambda: scenes.soft_cube_on_plane("rich", height=0.106),
            self.TOL,
            num_steps=6,
        )


CONTACT_PARAM_FIELDS = (
    "penalty_coefficient",
    "coulomb_friction_coefficient",
    "viscous_friction_coefficient",
    "normal_viscous_damping_coefficient",
)


class SoftContactParameterGradientTest(unittest.TestCase):
    """Contact-parameter and gravity gradients on the soft contact scene.

    The parameter adjoint re-assembles the island residual under perturbed
    parameters (-lambda^T dR/dtheta); with contact on a soft island that
    residual includes the async-contact term, so this validates that the
    soft contact assembly is exactly the one the state adjoint saw.
    Measured (2026-09-01, 'rich' regime): per-field 7.6e-7..1.3e-6, gravity
    2.2e-7 relative; tolerances keep >= 30x headroom.
    """

    def _adjoint_grads(self, scene, cube):
        num_nodes = cube.get_num_dofs() // 3
        diffsim.reset_back_propagation(scene)
        pre, post = [], []
        for _ in range(NUM_STEPS):
            pre.append(scene.capture_state())
            scene.step(DT)
            post.append(scene.capture_state())
        for step in reversed(range(NUM_STEPS)):
            diffsim.prepare_back_propagate(scene, post[step], pre[step])
            if step == NUM_STEPS - 1:
                diffsim.get_displacements_backward(cube, _loss_grad(cube, num_nodes))
            diffsim.back_propagate(scene)
        grad_cp = np.zeros(len(CONTACT_PARAM_FIELDS))
        diffsim.set_contact_params_backward(cube, grad_cp)
        grad_g = np.zeros(3)
        diffsim.set_gravity_backward(scene, grad_g)
        for handle in pre + post:
            scene.release_state(handle)
        return grad_cp, grad_g

    @staticmethod
    def _rollout_loss(gravity=None, field=None, value=None) -> float:
        scene, cube = scenes.soft_cube_on_plane("rich")
        try:
            _configure(scene)
            if gravity is not None:
                scene.set_gravity(gravity)
            if field is not None:
                params = cube.get_contact_params()
                setattr(params, field, value)
                cube.set_contact_params(params)
            for _ in range(NUM_STEPS):
                scene.step(DT)
            return _loss_value(cube, cube.get_num_dofs() // 3)
        finally:
            physics.destroy_scene(scene)

    def test_contact_params_and_gravity_vs_fd(self) -> None:
        scene, cube = scenes.soft_cube_on_plane("rich")
        self.addCleanup(physics.destroy_scene, scene)
        _configure(scene)
        base = {f: getattr(cube.get_contact_params(), f) for f in CONTACT_PARAM_FIELDS}
        gravity0 = np.asarray(scene.get_gravity(), dtype=np.float64)
        grad_cp, grad_g = self._adjoint_grads(scene, cube)

        eps = 1e-6
        for f_i, field in enumerate(CONTACT_PARAM_FIELDS):
            self.assertGreater(base[field], 0.0)  # central FD stays admissible
            h = eps * (1.0 + abs(base[field]))
            fd = (
                self._rollout_loss(field=field, value=base[field] + h)
                - self._rollout_loss(field=field, value=base[field] - h)
            ) / (2.0 * h)
            denom = max(abs(fd), abs(grad_cp[f_i]))
            self.assertGreater(denom, 1e-13, f"{field}: test is vacuous")
            self.assertLessEqual(
                abs(grad_cp[f_i] - fd) / denom,
                1e-4,
                f"{field}: adjoint={grad_cp[f_i]:.6e}, fd={fd:.6e}",
            )

        fd_g = np.zeros(3)
        for i in range(3):
            gp, gm = gravity0.copy(), gravity0.copy()
            gp[i] += eps
            gm[i] -= eps
            fd_g[i] = (
                self._rollout_loss(gravity=gp) - self._rollout_loss(gravity=gm)
            ) / (2.0 * eps)
        self.assertGreater(np.abs(fd_g).max(), 0.0, "test is vacuous")
        np.testing.assert_allclose(grad_g, fd_g, rtol=1e-5, atol=0.0)


def _tilted_soft_cube_on_plane(tilt_deg: float, max_alignment_normals: float):
    """Soft cube rotated about y so one bottom edge rests on the plane, with
    the alignment-fading slope amplified through max_alignment_normals."""
    scene = physics.create_scene("diffsim_soft_tilted")
    scene.set_gravity(scenes.GRAVITY)
    cp = scenes.contact_params("rich")
    cp.max_alignment_normals = max_alignment_normals
    scene.create_rigid_actor(
        name="ground",
        shape=physics.create_plane_shape(normal=[0, 0, 1], distance=0.0),
        is_static=True,
        contact=cp,
    )
    theta = np.deg2rad(tilt_deg)
    rotation = physics.Quaternion(0.0, np.sin(theta / 2.0), 0.0, np.cos(theta / 2.0))
    height = 0.1 * (np.cos(theta) + np.sin(theta)) - 0.001  # lowest corner 1 mm deep
    cube = scene.create_soft_actor(
        name="jelly",
        shape=scenes.cube_shape(),
        material=physics.SoftMaterialParams(),
        contact=cp,
        world_from_local=physics.TransformRT(rotation, [0.0, 0.0, height]),
    )
    cube.set_node_velocities_local(np.tile(np.asarray(V0), cube.get_num_dofs() // 3))
    return scene, cube


class SoftContactApproximationPinTest(unittest.TestCase):
    """Pins the one approximation of the contact adjoint (shared with the
    rigid path, which was measured under the identical probe).

    With explicit normals the dissipative terms use the colliding surface
    normal evaluated at the stage start, which depends on the previous state
    (nodal displacements here, the rotation for rigid bodies). Both the rigid
    and the soft GradTarget::Previous assemblies treat that normal as a
    constant. Where it enters - the alignment-fading factor (max_alignment -
    n_collider . n_colliding) / (max_alignment + 1) - the dropped term is
    zero for flat contact (the dot product is stationary at anti-alignment)
    and O(sin(tilt) / (max_alignment_normals + 1)) otherwise. Measured
    2026-09-01 on a 5-degree tilt: 2.4e-6 at the default fading (noise
    floor), 6.9e-5 with max_alignment_normals = -0.9, 6.8e-4 with -0.99
    (rigid path under the same probe: 8.2e-5 and 8.3e-4). The lower bound
    below flips this test when the term gets implemented.
    """

    def _rel_error(self, max_alignment_normals: float) -> float:
        factory = lambda: _tilted_soft_cube_on_plane(5.0, max_alignment_normals)
        scene, cube = factory()
        self.addCleanup(physics.destroy_scene, scene)
        _configure(scene)
        grad_u0, grad_v0, _, _, fd_valid = _adjoint_sweep(scene, cube, NUM_STEPS)
        self.assertTrue(fd_valid)
        fd_u0, fd_v0 = _fd_initial_grads(factory, NUM_STEPS)
        self.assertGreater(np.abs(fd_u0).max(), 0.0, "test is vacuous")

        def rel(a, b):
            return np.abs(a - b).max() / max(np.abs(a).max(), np.abs(b).max())

        return max(rel(grad_u0, fd_u0), rel(grad_v0, fd_v0))

    def test_tilted_contact_default_fading_agrees(self) -> None:
        self.assertLessEqual(self._rel_error(0.0), 1e-4)

    def test_tilted_contact_amplified_fading_shows_dropped_term(self) -> None:
        err = self._rel_error(-0.99)
        self.assertGreater(err, 1e-4, "dropped normal derivative no longer visible")
        self.assertLess(err, 3e-3)


if __name__ == "__main__":
    unittest.main()
