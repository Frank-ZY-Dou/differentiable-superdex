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

Material parameters (phase 5d): ``diffsim.set_soft_material_params_backward``
returns dL/d[youngs_modulus, poisson_ratio, density, mass_damping]. Ground
truth: rollout finite differences (relative steps, Richardson extrapolation
for Poisson's ratio) on a squashed free cube and on the contact scene, plus
two closed forms - (a) the free-body equation of motion depends on E and rho
only through E/rho (gravity is mass-independent), so E dL/dE + rho dL/drho =
0 exactly for any loss without contact; (b) an undeformed free-falling cube
has no E, nu or rho sensitivity at all. The mass-damping coefficient is
gated at zero, so at alpha = 0 the engine reports the right-sided derivative
and the test compares against a forward difference.

Loud-contract tests pin the remaining exclusions: sync (dynamic-dynamic)
contact involving a soft actor, stiffness damping, missing inertia, Dirichlet
BCs, the forced recentering disable, and the material-gradient scope
(homogeneous Lame-type materials only). (The DifferentiableRollout driver
and the torch bridge drive soft actors since 5c; see test_diffsim_rollout /
test_diffsim_torch.)

Requires SUPERDEX_PRECISION=double and a native build with the soft adjoint.
"""

from __future__ import annotations

import os
import unittest

import numpy as np
import superdex.physics as physics

from superdex.physics.diffsim_rollout import DifferentiableRollout
from superdex.physics.utils.penetration import PenetrationChecker

from . import scenes
from .harness import DisplacementErrorLoss, TranslationErrorLoss

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

    def test_dynamic_actor_in_the_same_island_is_allowed(self) -> None:
        # A rigid cube resting on the soft cube: sync contact of the soft's samples
        # against the box's SDF, in one island. Supported since 2026-09-02 (the
        # backward used to refuse any island shared with a dynamic actor); the
        # gradient itself is validated by SoftSyncContactGradientTest. Soft actors
        # created from Python carry no collider, so the unsupported direction (the
        # box's samples against a soft SDF, which has no stage-start Hessian) cannot
        # arise here; the engine rejects it in back_propagate.
        scene, cube = self._soft_on_plane_with_rigid((0.0, 0.0, 0.298))
        self.addCleanup(physics.destroy_scene, scene)
        grad = self._one_step_backward(scene, cube)
        self.assertGreater(np.abs(grad).max(), 0.0)

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


class SoftSyncContactGradientTest(unittest.TestCase):
    """Sync contact (dynamic-dynamic, one island): the soft cube's samples against a
    rigid box sliding on top of it (:func:`scenes.soft_cube_under_rigid`). The
    gradient of the box's final position w.r.t. its initial velocity and w.r.t.
    the soft's initial nodal velocities flows through the frictional contact in
    both directions (the box moves the soft, the soft's reaction moves the box).
    The previous-state assembly of deformable samples against a moving rigid
    collider uses the stage-start collider-space Jacobians and lever arms (C++
    MochiSoftRigidContact *Previous tests, 2026-09-02)."""

    dt, num_steps = DT, 3
    goal = np.array([0.03, 0.0, 0.3])

    class _BoxLoss:
        def __init__(self, rigid, goal):
            self.rigid, self.goal = rigid, np.asarray(goal, dtype=np.float64)

        def value(self) -> float:
            d = np.asarray(self.rigid.get_center_of_mass_transform().translation) - self.goal
            return 0.5 * float(d @ d)

        def accumulate_output_grad(self) -> None:
            d = np.asarray(self.rigid.get_center_of_mass_transform().translation) - self.goal
            g = np.zeros(7)
            g[:3] = d
            diffsim.get_center_of_mass_transform_backward(self.rigid, g)

    def _rollout_loss(self, rigid_velocity, soft_velocity_delta=None) -> float:
        scene, soft, rigid = scenes.soft_cube_under_rigid(rigid_velocity=tuple(rigid_velocity))
        try:
            _configure(scene)
            if soft_velocity_delta is not None:
                soft.set_node_velocities_local(soft_velocity_delta)
            loss = self._BoxLoss(rigid, self.goal)
            for _ in range(self.num_steps):
                scene.step(self.dt)
            return loss.value()
        finally:
            physics.destroy_scene(scene)

    def test_initial_velocity_gradients_through_sync_contact(self) -> None:
        v0_rigid = np.array([0.2, 0.0, 0.0])
        scene, soft, rigid = scenes.soft_cube_under_rigid(rigid_velocity=tuple(v0_rigid))
        self.addCleanup(physics.destroy_scene, scene)
        _configure(scene)
        result = DifferentiableRollout(scene, dt=self.dt, num_steps=self.num_steps).run(
            apply_inputs=lambda step: None, terminal_losses=[self._BoxLoss(rigid, self.goal)]
        )
        self.assertTrue(result.fd_valid, result.flagged_steps)
        self.assertEqual(result.minres_fallbacks, 0)
        grad_rigid = result.gradients["rigid"].initial_velocity[:3]
        grad_soft = result.gradients["jelly"].initial_velocity
        num_nodes = soft.get_num_dofs() // 3

        # Rigid initial velocity (norm-relative: the scene is symmetric in y).
        fd_rigid = np.zeros(3)
        for k in range(3):
            fds = []
            for eps in (1e-5, 1e-6):
                dv = np.zeros(3)
                dv[k] = eps
                fds.append(
                    (self._rollout_loss(v0_rigid + dv) - self._rollout_loss(v0_rigid - dv))
                    / (2.0 * eps)
                )
            fd_rigid[k] = fds[0]
            if abs(fds[0]) > 1e-8:
                self.assertLessEqual(abs(fds[0] - fds[1]) / abs(fds[0]), 1e-4, "rough FD")
        self.assertGreater(np.linalg.norm(fd_rigid), 0.0)
        rel = np.linalg.norm(grad_rigid - fd_rigid) / np.linalg.norm(fd_rigid)
        self.assertLessEqual(rel, 1e-5, (grad_rigid, fd_rigid))

        # Soft initial nodal velocities: the z-velocities of the top-face nodes (in
        # contact with the box) and one x-velocity; an entry counts only when the
        # two FD estimates agree to 1e-4 relative, and at least two must count.
        rest = scenes.CUBE_COORDS.reshape(-1, 3)
        top = [i for i in range(num_nodes) if rest[i, 2] > 0.0]
        entries = [3 * i + 2 for i in top[:3]] + [3 * top[0]]
        rel_errors, skipped = {}, {}
        for k in entries:
            fds = []
            for eps in (1e-5, 1e-6):
                dv = np.zeros(3 * num_nodes)
                dv[k] = eps
                fds.append(
                    (self._rollout_loss(v0_rigid, dv) - self._rollout_loss(v0_rigid, -dv))
                    / (2.0 * eps)
                )
            denom = max(abs(fds[0]), 1e-30)
            fd_self = abs(fds[0] - fds[1]) / denom
            if fd_self > 1e-4:
                skipped[k] = fd_self
                continue
            rel_errors[k] = abs(grad_soft[k] - fds[0]) / denom
        self.assertGreaterEqual(len(rel_errors), 2, f"too few smooth entries: {skipped}")
        self.assertLessEqual(max(rel_errors.values()), 1e-4, (rel_errors, skipped))


class DeformableColliderGradientTest(unittest.TestCase):
    """A soft cube as the COLLIDER of other actors' samples (an SDF collider created through
    the experimental API: its rest-space grid SDF mapped through the deformation). The
    previous-state assembly differentiates the stage-start contact data through the Jacobians
    of the collider's stage-start mapping (``jacWorldFromDofsStageStart``, 2026-09-05); with
    the current mapping in their place the rigid-box gradients below were 1e-3 relative off
    finite differences, and the soft-on-soft case aborted."""

    dt = DT

    def _box_loss_rollout(self, friction, forces, num_steps):
        """Loss of the box's final position under per-step external forces, fresh scene."""
        scene, soft, rigid = scenes.rigid_on_soft_collider(friction)
        try:
            _configure(scene)
            loss = TranslationErrorLoss(rigid, ref=np.array([0.08, 0.0, 0.29]))
            dofs = np.arange(6, dtype=np.int32)
            for step in range(num_steps):
                rigid.set_external_forces_on_dofs(dofs, np.ascontiguousarray(forces[step]))
                scene.step(self.dt)
            return loss.value()
        finally:
            physics.destroy_scene(scene)

    def test_rigid_box_on_soft_collider_force_gradients(self) -> None:
        """The box's samples against the soft cube's mapped SDF and the soft's samples
        against the box's SDF, in one island: the gradient of the box's final position with
        respect to its per-step external forces (all six DoFs at the first, a middle and the
        last step) against central FD at eps 1e-6, to 1e-4 relative plus 1e-11 absolute (the
        loss is of order 1e-3, so the difference quotients carry about 1e-12 of rounding,
        which dominates the entries near 1e-8 that the scene's symmetry in y leaves; the FD
        estimates at eps 1e-6 and 1e-7 agree to 1e-4 for every entry above 1e-6). Measured
        (2026-09-05): 2.5e-6 for the x-force at the first step, 1.3e-9 for the y-torque, at
        most 1.0e-4 on a 2e-8 entry."""
        num_steps = 8
        forces = np.zeros((num_steps, 6))
        forces[:, 0] = np.linspace(2.0, 6.0, num_steps)
        forces[:, 4] = 0.3
        scene, soft, rigid = scenes.rigid_on_soft_collider("coulomb")
        self.addCleanup(physics.destroy_scene, scene)
        _configure(scene)
        checker = PenetrationChecker(scene)
        loss = TranslationErrorLoss(rigid, ref=np.array([0.08, 0.0, 0.29]))
        dofs = np.arange(6, dtype=np.int32)

        def apply(step):
            rigid.set_external_forces_on_dofs(dofs, np.ascontiguousarray(forces[step]))
            if step > 0:
                checker.record(step)

        result = DifferentiableRollout(scene, dt=self.dt, num_steps=num_steps).run(
            apply_inputs=apply, terminal_losses=[loss]
        )
        self.assertTrue(result.fd_valid, result.flagged_steps)
        self.assertIn("jelly / rigid", checker.report(4), "the box and the soft cube must touch")
        grad = result.gradients["rigid"].external_forces
        self.assertAlmostEqual(self._box_loss_rollout("coulomb", forces, num_steps), result.loss, delta=1e-12)
        adjoint, fd = [], []
        for step in (0, num_steps // 2, num_steps - 1):
            for dof in range(6):
                fds = []
                for eps in (1e-6, 1e-7):
                    pair = []
                    for sign in (+1.0, -1.0):
                        perturbed = forces.copy()
                        perturbed[step, dof] += sign * eps
                        pair.append(self._box_loss_rollout("coulomb", perturbed, num_steps))
                    fds.append((pair[0] - pair[1]) / (2.0 * eps))
                if abs(fds[0]) > 1e-6:
                    self.assertLessEqual(abs(fds[0] - fds[1]) / abs(fds[0]), 1e-4, (step, dof, fds))
                adjoint.append(grad[dof, step])
                fd.append(fds[0])
        adjoint, fd = np.array(adjoint), np.array(fd)
        self.assertGreater(float(np.abs(fd).max()), 1e-6, "vacuous: the forces do not move the loss")
        np.testing.assert_allclose(adjoint, fd, rtol=1e-4, atol=1e-11)

    def test_contact_force_adjoint_against_a_deformable_collider_runs(self) -> None:
        """The box's total contact force depends on the soft collider's nodal positions; the
        contact-force adjoint reaches them (exactness: EngineContactForceAdjointTest in
        test_diffsim_torch). Here: the adjoint runs and leaves a nonzero gradient on the
        soft cube's initial nodal velocities."""
        scene, soft, rigid = scenes.rigid_on_soft_collider("coulomb")
        self.addCleanup(physics.destroy_scene, scene)
        _configure(scene)
        rigid.register_query(physics.QueryType.TOTAL_CONTACT_FORCE)

        class ForceLoss:
            def value(self) -> float:
                f = np.asarray(rigid.get_contact_force_world(), dtype=np.float64)
                return 0.5 * float(f @ f)

            def accumulate_output_grad(self) -> None:
                diffsim.get_contact_force_world_backward(
                    rigid, np.asarray(rigid.get_contact_force_world(), dtype=np.float64)
                )

        result = DifferentiableRollout(scene, dt=self.dt, num_steps=4).run(
            apply_inputs=lambda step: None, terminal_losses=[ForceLoss()]
        )
        self.assertTrue(result.fd_valid, result.flagged_steps)
        self.assertGreater(float(np.abs(result.gradients["jelly"].initial_velocity).max()), 0.0)

    def _drop_loss_rollout(self, forces, num_steps):
        """Loss of the box's final position when dropped onto the soft collider from 5 cm above
        it at 1 m/s, under per-step external forces; fresh scene."""
        scene, soft, rigid = scenes.rigid_on_soft_collider("coulomb")
        try:
            _configure(scene)
            rigid.set_center_of_mass_transform(physics.TransformRT([0.03, 0.0, 0.35]))
            rigid.set_velocity([0.0, 0.0, -1.0], [0.0, 0.0, 0.0])
            loss = TranslationErrorLoss(rigid, ref=np.array([0.05, 0.0, 0.29]))
            dofs = np.arange(6, dtype=np.int32)
            for step in range(num_steps):
                rigid.set_external_forces_on_dofs(dofs, np.ascontiguousarray(forces[step]))
                scene.step(self.dt)
            return loss.value()
        finally:
            physics.destroy_scene(scene)

    def test_contact_onset_on_a_soft_collider(self) -> None:
        """A box dropped onto the soft collider (5 cm above it at 1 m/s, landing during the
        fifth step): the contact samples that map into the collider at the current state but
        not at the stage start of the step get filled-in stage-start data from the current
        state. Until 2026-09-07 that fill asserted on the current SDF Hessians, which a forward
        step of a differentiable scene does not compute (before 2026-09-07: the
        process aborted at the fifth step; the resting-contact fixtures never arrive). The fill
        now sets a zero stage-start Hessian for those contacts (the filled normal is the current
        one and does not depend on the stage-start position). The forward runs, and the
        gradients of the box's final position w.r.t. its per-step forces before, during and
        after the onset agree with central FD at eps 1e-6 (self-consistent with eps 1e-7):
        measured 1e-6 to 4e-5 relative on the entries above 1e-7, and within 8e-11 absolute
        below (the difference quotients of this loss of order 1e-2 carry about 1e-10 of
        rounding at eps 1e-6, which is what the near-zero entries the scene's symmetry leaves
        compare against)."""
        num_steps = 10
        forces = np.zeros((num_steps, 6))
        forces[:, 0] = np.linspace(1.0, 4.0, num_steps)
        forces[:, 4] = 0.2
        scene, soft, rigid = scenes.rigid_on_soft_collider("coulomb")
        self.addCleanup(physics.destroy_scene, scene)
        _configure(scene)
        rigid.set_center_of_mass_transform(physics.TransformRT([0.03, 0.0, 0.35]))
        rigid.set_velocity([0.0, 0.0, -1.0], [0.0, 0.0, 0.0])
        checker = PenetrationChecker(scene)
        loss = TranslationErrorLoss(rigid, ref=np.array([0.05, 0.0, 0.29]))
        dofs = np.arange(6, dtype=np.int32)

        def apply(step):
            rigid.set_external_forces_on_dofs(dofs, np.ascontiguousarray(forces[step]))
            if step > 0:
                checker.record(step)

        result = DifferentiableRollout(scene, dt=self.dt, num_steps=num_steps).run(
            apply_inputs=apply, terminal_losses=[loss]
        )
        self.assertTrue(result.fd_valid, result.flagged_steps)
        self.assertIn("jelly / rigid", checker.report(4), "the box must land on the soft cube")
        self.assertAlmostEqual(self._drop_loss_rollout(forces, num_steps), result.loss, delta=1e-12)
        grad = result.gradients["rigid"].external_forces
        adjoint, fd = [], []
        for step in (0, 4, 6, num_steps - 1):
            for dof in range(6):
                fds = []
                for eps in (1e-6, 1e-7):
                    pair = []
                    for sign in (+1.0, -1.0):
                        perturbed = forces.copy()
                        perturbed[step, dof] += sign * eps
                        pair.append(self._drop_loss_rollout(perturbed, num_steps))
                    fds.append((pair[0] - pair[1]) / (2.0 * eps))
                if abs(fds[0]) > 1e-6:
                    # The same tolerance as the comparison below: with worker threads the
                    # reductions of the two rollouts of a difference quotient are not ordered
                    # alike, which is a few 1e-10 on these quotients.
                    self.assertLessEqual(abs(fds[0] - fds[1]), 1e-4 * abs(fds[0]) + 1e-9, (step, dof, fds))
                adjoint.append(grad[dof, step])
                fd.append(fds[0])
        adjoint, fd = np.array(adjoint), np.array(fd)
        self.assertGreater(float(np.abs(fd).max()), 1e-6, "vacuous: the forces do not move the loss")
        np.testing.assert_allclose(adjoint, fd, rtol=1e-4, atol=1e-9)

    def _stack_loss_rollout(self, friction, v_top, v_bottom, num_steps):
        scene, bottom, top = scenes.soft_on_soft(friction)
        try:
            _configure(scene)
            top.set_node_velocities_local(np.ascontiguousarray(v_top))
            bottom.set_node_velocities_local(np.ascontiguousarray(v_bottom))
            loss = DisplacementErrorLoss(top, ref=(0.02, 0.0, -0.01))
            for _ in range(num_steps):
                scene.step(self.dt)
            return loss.value()
        finally:
            physics.destroy_scene(scene)

    def test_soft_on_soft_initial_velocity_gradients(self) -> None:
        """Two stacked soft cubes, each the collider of the other's samples: the gradient of
        the top cube's displacements with respect to the initial nodal velocities of both
        cubes, along two random directions each, against central FD at eps 1e-6 / 1e-7.
        Measured 5e-9 to 2e-7."""
        num_steps = 8
        scene, bottom, top = scenes.soft_on_soft("coulomb")
        self.addCleanup(physics.destroy_scene, scene)
        _configure(scene)
        n_top, n_bottom = top.get_num_dofs(), bottom.get_num_dofs()
        v_top = np.zeros(n_top)
        v_top[0::3] = 0.2
        v_bottom = np.zeros(n_bottom)
        top.set_node_velocities_local(np.ascontiguousarray(v_top))
        checker = PenetrationChecker(scene)
        loss = DisplacementErrorLoss(top, ref=(0.02, 0.0, -0.01))
        result = DifferentiableRollout(scene, dt=self.dt, num_steps=num_steps).run(
            apply_inputs=lambda step: checker.record(step) if step > 0 else None,
            terminal_losses=[loss],
        )
        self.assertTrue(result.fd_valid, result.flagged_steps)
        self.assertIn("bottom / top", checker.report(4), "the cubes must touch")
        self.assertAlmostEqual(
            self._stack_loss_rollout("coulomb", v_top, v_bottom, num_steps), result.loss, delta=1e-12
        )
        rng = np.random.default_rng(7)
        for name, n, base_top, base_bottom in (
            ("top", n_top, True, False),
            ("bottom", n_bottom, False, True),
        ):
            grad = result.gradients[name].initial_velocity
            self.assertGreater(float(np.abs(grad).max()), 0.0, f"{name}: vacuous")
            for _ in range(2):
                d = rng.standard_normal(n)
                d /= np.linalg.norm(d)
                fds = []
                for eps in (1e-6, 1e-7):
                    pair = []
                    for sign in (+1.0, -1.0):
                        pair.append(
                            self._stack_loss_rollout(
                                "coulomb",
                                v_top + sign * eps * d if base_top else v_top,
                                v_bottom + sign * eps * d if base_bottom else v_bottom,
                                num_steps,
                            )
                        )
                    fds.append((pair[0] - pair[1]) / (2.0 * eps))
                self.assertLessEqual(abs(fds[0] - fds[1]) / abs(fds[0]), 1e-4, (name, fds))
                self.assertLessEqual(abs(float(grad @ d) - fds[0]) / abs(fds[0]), 1e-5, (name, grad @ d, fds))


class SoftArticulatedContactGradientTest(unittest.TestCase):
    """A controlled chain pushing a soft cube along the ground
    (:func:`scenes.chain_pushing_soft_cube`): the gradient of the cube's mean
    displacement w.r.t. the chain's per-step joint targets flows through the
    sync contact of the soft's samples against the link's SDF (an articulated
    collider: reduced-coordinate collider Jacobians at the stage-start state)."""

    dt, num_steps = 0.01, 30
    goal = np.array([0.08, 0.0, -0.01])

    @classmethod
    def controls(cls):
        return np.stack(
            [np.linspace(0.0, -1.2, cls.num_steps), np.zeros(cls.num_steps)], axis=1
        )

    class _MeanDisplacementLoss:
        def __init__(self, soft, goal):
            self.soft, self.goal = soft, np.asarray(goal, dtype=np.float64)
            self.num_nodes = soft.get_num_dofs() // 3

        def _mean(self) -> np.ndarray:
            return np.asarray(self.soft.get_displacements()).reshape(-1, 3).mean(axis=0)

        def value(self) -> float:
            d = self._mean() - self.goal
            return 0.5 * float(d @ d)

        def accumulate_output_grad(self) -> None:
            d = self._mean() - self.goal
            diffsim.get_displacements_backward(self.soft, np.tile(d / self.num_nodes, self.num_nodes))

    def _rollout_loss(self, controls) -> float:
        scene, chain, soft = scenes.chain_pushing_soft_cube()
        try:
            _configure(scene)
            loss = self._MeanDisplacementLoss(soft, self.goal)
            for step in range(self.num_steps):
                chain.set_articulated_target_pose(np.ascontiguousarray(controls[step]))
                scene.step(self.dt)
            return loss.value()
        finally:
            physics.destroy_scene(scene)

    def test_control_gradients_through_soft_link_contact(self) -> None:
        controls = self.controls()
        scene, chain, soft = scenes.chain_pushing_soft_cube()
        self.addCleanup(physics.destroy_scene, scene)
        _configure(scene)
        result = DifferentiableRollout(scene, dt=self.dt, num_steps=self.num_steps).run(
            apply_inputs=lambda step: chain.set_articulated_target_pose(
                np.ascontiguousarray(controls[step])
            ),
            terminal_losses=[self._MeanDisplacementLoss(soft, self.goal)],
        )
        self.assertTrue(result.fd_valid, result.flagged_steps)
        self.assertEqual(result.minres_fallbacks, 0)
        grad = result.gradients["chain"].control_targets  # (2, num_steps)
        self.assertGreater(np.linalg.norm(grad), 0.0, "the cube was not pushed")
        # Directional derivative along the gradient (the clean check), then the
        # largest entries at their own finite-difference self-consistency level.
        direction = grad / np.linalg.norm(grad)
        fds = [
            (self._rollout_loss(controls + eps * direction.T)
             - self._rollout_loss(controls - eps * direction.T)) / (2.0 * eps)
            for eps in (1e-5, 1e-6)
        ]
        self.assertLessEqual(abs(fds[0] - fds[1]) / abs(fds[0]), 1e-4, "rough directional FD")
        self.assertLessEqual(abs(np.linalg.norm(grad) - fds[0]) / abs(fds[0]), 1e-4, fds)
        rel_errors, skipped = {}, {}
        for flat in np.argsort(-np.abs(grad).ravel())[:4]:
            d, s = divmod(int(flat), self.num_steps)
            fds = []
            for eps in (1e-5, 1e-6):
                plus, minus = controls.copy(), controls.copy()
                plus[s, d] += eps
                minus[s, d] -= eps
                fds.append((self._rollout_loss(plus) - self._rollout_loss(minus)) / (2.0 * eps))
            denom = max(abs(fds[0]), 1e-30)
            fd_self = abs(fds[0] - fds[1]) / denom
            if fd_self > 1e-3:
                skipped[(d, s)] = fd_self
                continue
            rel_errors[(d, s)] = abs(grad[d, s] - fds[0]) / denom
        self.assertGreaterEqual(len(rel_errors), 3, f"too few smooth entries: {skipped}")
        self.assertLessEqual(max(rel_errors.values()), 1e-3, (rel_errors, skipped))


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
    """Guards the removal of the two approximations of the contact adjoint
    (shared with the rigid path, which was measured under the identical
    probe).

    With explicit normals the dissipative terms use normals evaluated at the
    stage start, which depend on the previous state (nodal displacements
    here, the rotation for rigid bodies). The GradTarget::Previous assemblies
    used to treat them as constants. (1) The colliding surface normal enters
    only through the alignment-fading factor (max_alignment - n_collider .
    n_colliding) / (max_alignment + 1), whose dropped derivative is zero for
    flat contact (the dot product is stationary at anti-alignment) and
    O(sin(tilt) / (max_alignment_normals + 1)) otherwise. Measured 2026-09-01
    on a 5-degree tilt with fading on: 2.4e-6 at the default fading (noise
    floor), 6.9e-5 with max_alignment_normals = -0.9, 6.8e-4 with -0.99
    (rigid path under the same probe: 8.2e-5 and 8.3e-4); on rotating rigid
    bodies the same term reaches 1e-2..1e1. Since 2026-09-01 differentiable
    scenes force fade_friction = false (mochi_scene.cpp), which removes the
    factor and with it the dropped term: the amplified configuration measures
    1.1e-8, and the amplified test asserts exactness so that it flips if the
    override is ever removed. (2) The collider normal (normalized stage-start
    SDF gradient) is now differentiated with the stage-start SDF Hessian
    (contact_utils.h); on the plane used here that Hessian is zero, the
    rigid-body edge-region case is covered by
    test_diffsim_gradients.test_frictional_contact_through_sdf_edge_regions_is_exact.
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

    def test_tilted_contact_amplified_fading_is_exact_with_fading_off(self) -> None:
        scene, _ = _tilted_soft_cube_on_plane(5.0, -0.99)
        self.addCleanup(physics.destroy_scene, scene)
        _configure(scene)
        self.assertFalse(
            scene.get_solver_params().experimental_eval.fade_friction,
            "differentiable scenes must run with fade_friction off",
        )
        self.assertLessEqual(self._rel_error(-0.99), 1e-4)


MATERIAL_FIELDS = (
    "youngs_modulus",
    "poisson_ratio",
    "density",
    "mass_damping_coefficient",
)


def _material_get(params, field: str) -> float:
    if field in ("youngs_modulus", "poisson_ratio"):
        return float(getattr(params.neo_hookean, field))
    return float(getattr(params, field))


def _material_set(params, field: str, value: float) -> None:
    if field in ("youngs_modulus", "poisson_ratio"):
        nh = params.neo_hookean
        setattr(nh, field, value)
        params.neo_hookean = nh
    else:
        setattr(params, field, value)


def _material_adjoint(scene, cube, num_steps: int) -> np.ndarray:
    """Forward + reverse sweep with the terminal loss; returns dL/d(material)."""
    num_nodes = cube.get_num_dofs() // 3
    diffsim.reset_back_propagation(scene)
    pre, post = [], []
    for _ in range(num_steps):
        pre.append(scene.capture_state())
        scene.step(DT)
        post.append(scene.capture_state())
    for step in reversed(range(num_steps)):
        diffsim.prepare_back_propagate(scene, post[step], pre[step])
        if step == num_steps - 1:
            diffsim.get_displacements_backward(cube, _loss_grad(cube, num_nodes))
        diffsim.back_propagate(scene)
    grad = np.zeros(len(MATERIAL_FIELDS))
    diffsim.set_soft_material_params_backward(cube, grad)
    for handle in pre + post:
        scene.release_state(handle)
    return grad


def _material_fd(scene_factory, num_steps: int, base: dict) -> np.ndarray:
    """Independent ground truth: finite differences of the rollout loss over
    ``set_soft_material_params``. Relative steps of 1e-4 (the loss is far less
    sensitive to E than to the state, so a 1e-6 step would be dominated by
    cancellation); Richardson-extrapolated central differences for Poisson's
    ratio (step 1e-2 of the distance to the nearer pole); a forward
    difference for the mass-damping coefficient at zero, where the engine's
    non-negativity check forbids the negative side."""

    def rollout_loss(field=None, value=None) -> float:
        scene, cube = scene_factory()
        try:
            _configure(scene)
            if field is not None:
                params = cube.get_soft_material_params()
                _material_set(params, field, value)
                cube.set_soft_material_params(params)
            for _ in range(num_steps):
                scene.step(DT)
            return _loss_value(cube, cube.get_num_dofs() // 3)
        finally:
            physics.destroy_scene(scene)

    fd = np.zeros(len(MATERIAL_FIELDS))
    for i, field in enumerate(MATERIAL_FIELDS):
        value = base[field]
        if field == "poisson_ratio":
            h = 1e-2 * min(0.5 - value, 1.0 + value)

            def central(step):
                return (
                    rollout_loss(field, value + step) - rollout_loss(field, value - step)
                ) / (2.0 * step)

            fd[i] = (4.0 * central(h / 2.0) - central(h)) / 3.0
            continue
        h = 1e-4 * (1.0 + abs(value))
        if field == "mass_damping_coefficient" and value <= 0.0:
            fd[i] = (rollout_loss(field, value + h) - rollout_loss()) / h
        else:
            fd[i] = (rollout_loss(field, value + h) - rollout_loss(field, value - h)) / (
                2.0 * h
            )
    return fd


class SoftMaterialGradientTest(unittest.TestCase):
    """dL/d(Young's modulus, Poisson's ratio, density, mass damping).

    Measured on 2026-09-01 (3 steps): per-field agreement with the rollout
    FD between 2e-9 and 1.7e-6 relative on the squashed free cube (with and
    without mass damping) and on the contact scene; the E/rho homogeneity
    invariant holds to 1e-10 (3 steps) and 1e-7 (8 steps). Tolerances keep
    >= 50x headroom.
    """

    def _compare(self, adjoint, fd, tol: float) -> None:
        for i, field in enumerate(MATERIAL_FIELDS):
            denom = max(abs(adjoint[i]), abs(fd[i]))
            self.assertGreater(denom, 1e-13, f"{field}: test is vacuous")
            self.assertLessEqual(
                abs(adjoint[i] - fd[i]) / denom,
                tol,
                f"{field}: adjoint={adjoint[i]:.6e}, fd={fd[i]:.6e}",
            )

    def _run(self, scene_factory, num_steps: int = NUM_STEPS):
        scene, cube = scene_factory()
        self.addCleanup(physics.destroy_scene, scene)
        _configure(scene)
        base = {f: _material_get(cube.get_soft_material_params(), f) for f in MATERIAL_FIELDS}
        adjoint = _material_adjoint(scene, cube, num_steps)
        return adjoint, base

    def test_squashed_free_cube_vs_fd_and_homogeneity(self) -> None:
        factory = lambda: scenes.soft_cube(mass_damping=2.0, squash=0.1)
        adjoint, base = self._run(factory)
        fd = _material_fd(factory, NUM_STEPS, base)
        self._compare(adjoint, fd, 1e-4)
        # Closed form: without contact the motion depends on E and rho only
        # through E/rho, so the gradients are homogeneous of degree zero in
        # (E, rho): E dL/dE + rho dL/drho = 0.
        e_term = base["youngs_modulus"] * adjoint[0]
        rho_term = base["density"] * adjoint[2]
        self.assertGreater(abs(e_term), 0.0, "test is vacuous")
        self.assertLessEqual(abs(e_term + rho_term), 1e-8 * max(abs(e_term), abs(rho_term)))

    def test_homogeneity_holds_over_long_rollout(self) -> None:
        adjoint, base = self._run(
            lambda: scenes.soft_cube(mass_damping=2.0, squash=0.1), num_steps=8
        )
        e_term = base["youngs_modulus"] * adjoint[0]
        rho_term = base["density"] * adjoint[2]
        self.assertGreater(abs(e_term), 0.0, "test is vacuous")
        self.assertLessEqual(abs(e_term + rho_term), 1e-5 * max(abs(e_term), abs(rho_term)))

    def test_mass_damping_at_zero_is_the_right_derivative(self) -> None:
        # alpha = 0 sits on the engine's gate (alpha > 0 enables the term):
        # the reported value must be the one-sided derivative, matching a
        # forward difference, and be as large as with damping switched on.
        factory = lambda: scenes.soft_cube(squash=0.1)
        adjoint, base = self._run(factory)
        self.assertEqual(base["mass_damping_coefficient"], 0.0)
        fd = _material_fd(factory, NUM_STEPS, base)
        self._compare(adjoint, fd, 1e-4)
        self.assertGreater(abs(adjoint[3]), 1e-6)

    def test_undeformed_free_fall_has_no_elastic_or_density_sensitivity(self) -> None:
        # Rigid translation of an undeformed body: E, nu and rho cannot matter
        # (mass cancels under gravity), while mass damping still does. The
        # residual differences behind the E/nu/rho entries are pure round-off
        # here (measured 4e-16 against a damping gradient of 3e-5).
        adjoint, _base = self._run(lambda: scenes.soft_cube())
        self.assertGreater(abs(adjoint[3]), 1e-6, "test is vacuous")
        self.assertLessEqual(np.abs(adjoint[:3]).max(), 1e-9 * abs(adjoint[3]))

    def test_contact_scene_vs_fd(self) -> None:
        factory = lambda: scenes.soft_cube_on_plane("rich", mass_damping=1.0)
        adjoint, base = self._run(factory)
        fd = _material_fd(factory, NUM_STEPS, base)
        self._compare(adjoint, fd, 1e-4)
        # Contact forces are density-independent, so the homogeneity closed
        # form must NOT hold here (it is a free-body property).
        e_term = base["youngs_modulus"] * adjoint[0]
        rho_term = base["density"] * adjoint[2]
        self.assertGreater(abs(e_term + rho_term), 1e-2 * max(abs(e_term), abs(rho_term)))

    def test_contract_errors_and_reset(self) -> None:
        # Rigid actors have no soft material.
        scene, cube = scenes.rigid_free()
        self.addCleanup(physics.destroy_scene, scene)
        _configure(scene)
        with self.assertRaisesRegex(physics.Error, "soft actors"):
            diffsim.set_soft_material_params_backward(cube, np.zeros(4))

        # Read before any sweep.
        scene, jelly = scenes.soft_cube(squash=0.1)
        self.addCleanup(physics.destroy_scene, scene)
        _configure(scene)
        diffsim.reset_back_propagation(scene)
        with self.assertRaisesRegex(physics.Error, "not part of a back-propagated island"):
            diffsim.set_soft_material_params_backward(jelly, np.zeros(4))
        with self.assertRaisesRegex(physics.Error, "size must be 4"):
            diffsim.set_soft_material_params_backward(jelly, np.zeros(3))

        # Reset zeroes an accumulated gradient.
        grad = _material_adjoint(scene, jelly, NUM_STEPS)
        self.assertGreater(np.abs(grad).max(), 0.0)
        diffsim.reset_back_propagation(scene)
        zeroed = np.ones(4)
        diffsim.set_soft_material_params_backward(jelly, zeroed)
        np.testing.assert_array_equal(zeroed, np.zeros(4))

        # A per-element material field (heterogeneous parameters) is out of scope.
        field_params = jelly.get_soft_material_params()
        _material_set(field_params, "youngs_modulus", 2.0e5)
        physics.experimental.set_soft_material_params_field(jelly, field_params, 0)
        with self.assertRaisesRegex(physics.Error, "per-element material field"):
            diffsim.set_soft_material_params_backward(jelly, np.zeros(4))

    def test_non_lame_material_rejected(self) -> None:
        scene = physics.create_scene("soft_arap")
        self.addCleanup(physics.destroy_scene, scene)
        scene.set_gravity(scenes.GRAVITY)
        cube = scene.create_soft_actor(
            name="jelly",
            shape=scenes.cube_shape(),
            material=physics.SoftMaterialParams(
                type=physics.SoftMaterialType.ARAP,
                arap=physics.ArapMaterialParams(stiffness=1.0e5),
            ),
        )
        _configure(scene)
        num_nodes = cube.get_num_dofs() // 3
        diffsim.reset_back_propagation(scene)
        pre = scene.capture_state()
        scene.step(DT)
        post = scene.capture_state()
        diffsim.prepare_back_propagate(scene, post, pre)
        diffsim.get_displacements_backward(cube, _loss_grad(cube, num_nodes))
        diffsim.back_propagate(scene)
        with self.assertRaisesRegex(physics.Error, "Lame-type"):
            diffsim.set_soft_material_params_backward(cube, np.zeros(4))
        scene.release_all_states()


if __name__ == "__main__":
    unittest.main()
