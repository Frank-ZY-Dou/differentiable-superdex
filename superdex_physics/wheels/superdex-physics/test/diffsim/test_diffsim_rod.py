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

"""Discrete adjoints of rod actors (open elastic rods with a twist DoF per node).

What is differentiated (2026-09-02): the rod's inertia (translational and twist),
axial and bend/twist stresses, mass damping, gravity, constraints to rigid or
articulated actors, and centerline contact against static colliders. The material
frame axes are a carried state: they are parallel-transported from the previous
step and re-twisted, so the previous-state assembly differentiates the bend/twist
energy through that transport (the twist gauge and the holonomy w.r.t. the
previous tangents), and the end frame's dependence on the start frame at fixed
DoFs is carried explicitly. Differentiable rods retract poses from the stage-start
pose and carry a residual that is the exact gradient of the step energy in that
chart (a differentiable-scene-only change of the forward; see rod::AssembleBody).

Every gradient below is checked against independent rollout central finite
differences at two step sizes, counting an entry only where the two agree
(self-consistency), with the same protocol as test_diffsim_gradients.

Not supported (rejected by make_scene_differentiable or back_propagate): closed
loops, stiffness damping, user Dirichlet boundary conditions, contact skins,
rods as colliders, and contact with dynamic colliders.
"""

from __future__ import annotations

import os
import unittest

import numpy as np
import superdex.physics as physics
from superdex.physics.diffsim_rollout import DifferentiableRollout

from . import scenes
from .harness import configure_for_differentiability, diffsim

_NUM_WORKER_THREADS = int(os.environ.get("SUPERDEX_DIFFSIM_TEST_THREADS", "0"))


def setUpModule() -> None:
    if not physics.uses_double_precision():
        raise unittest.SkipTest("rod adjoint tests require SUPERDEX_PRECISION=double")
    physics.initialize(num_worker_threads=_NUM_WORKER_THREADS)


def tearDownModule() -> None:
    if physics.is_initialized():
        physics.shutdown()


def _node_position(rod, node: int) -> np.ndarray:
    ref = np.asarray(rod.get_mesh().coordinates).reshape(-1, 3)[node]
    return ref + np.asarray(rod.get_displacements())[4 * node : 4 * node + 3]


class _TipLoss:
    """0.5 |x_node - goal|^2 on a rod node."""

    def __init__(self, rod, node: int, goal):
        self.rod, self.node, self.goal = rod, node, np.asarray(goal, dtype=np.float64)

    def value(self) -> float:
        d = _node_position(self.rod, self.node) - self.goal
        return 0.5 * float(d @ d)

    def accumulate_output_grad(self) -> None:
        d = _node_position(self.rod, self.node) - self.goal
        g = np.zeros(self.rod.get_num_dofs())
        g[4 * self.node : 4 * self.node + 3] = d
        diffsim.get_displacements_backward(self.rod, g)


class _CubeLoss:
    def __init__(self, cube, goal):
        self.cube, self.goal = cube, np.asarray(goal, dtype=np.float64)

    def value(self) -> float:
        d = np.asarray(self.cube.get_center_of_mass_transform().translation) - self.goal
        return 0.5 * float(d @ d)

    def accumulate_output_grad(self) -> None:
        d = np.asarray(self.cube.get_center_of_mass_transform().translation) - self.goal
        g = np.zeros(7)
        g[:3] = d
        diffsim.get_center_of_mass_transform_backward(self.cube, g)


def _initial_velocity_check(test, build, loss_of, v0, num_steps, dt, entries, tol):
    """Adjoint dL/dv0 (rod nodal velocities, 4 per node) vs rollout central FD at
    eps 1e-5 and 1e-6; an entry counts only when the two FD estimates agree to
    1e-4 relative, and at least two entries must count."""
    scene, actors = build()
    test.addCleanup(physics.destroy_scene, scene)
    configure_for_differentiability(scene)
    rod = actors[0]
    rod.set_node_velocities_local(v0)
    result = DifferentiableRollout(scene, dt=dt, num_steps=num_steps).run(
        apply_inputs=lambda step: None, terminal_losses=[loss_of(actors)]
    )
    test.assertTrue(result.fd_valid, result.flagged_steps)
    grad = result.gradients[rod.get_name()].initial_velocity

    def rollout_loss(v):
        sc, acts = build()
        try:
            configure_for_differentiability(sc)
            acts[0].set_node_velocities_local(v)
            loss = loss_of(acts)
            for _ in range(num_steps):
                sc.step(dt)
            return loss.value()
        finally:
            physics.destroy_scene(sc)

    rel_errors, skipped = {}, {}
    for k in entries:
        fds = []
        for eps in (1e-5, 1e-6):
            dv = np.zeros_like(v0)
            dv[k] = eps
            fds.append((rollout_loss(v0 + dv) - rollout_loss(v0 - dv)) / (2.0 * eps))
        denom = max(abs(fds[0]), 1e-30)
        fd_self = abs(fds[0] - fds[1]) / denom
        if fd_self > 1e-4:
            skipped[k] = fd_self
            continue
        rel_errors[k] = abs(grad[k] - fds[0]) / denom
    test.assertGreaterEqual(len(rel_errors), 2, f"too few smooth entries: skipped {skipped}")
    test.assertLessEqual(max(rel_errors.values()), tol, (rel_errors, skipped))
    return rel_errors


class RodFreeAdjointTest(unittest.TestCase):
    """A free rod with a curved rest shape released under gravity with a velocity
    field that bends and twists it: inertia (incl. twist), axial and bend/twist
    stresses, and the frame transport between steps. Measured 2026-09-02:
    displacement-velocity entries 2e-8..3e-7, twist-rate entries 3e-7..8e-6
    (with FD self-consistency 2e-5), over 20 steps of 5 ms."""

    def test_initial_velocity_gradients(self) -> None:
        def build():
            scene, rod = scenes.rod_free()
            return scene, (rod,)

        scene, (rod,) = build()
        n = rod.get_num_dofs()
        num_nodes = n // 4
        physics.destroy_scene(scene)
        s = np.linspace(0.0, 0.4, num_nodes)
        v0 = np.zeros(n)
        for i in range(num_nodes):
            v0[4 * i : 4 * i + 3] = [0.2, 0.0, 0.1 + 0.8 * (s[i] / 0.4)]
            v0[4 * i + 3] = 3.0 * (1.0 - s[i] / 0.4)
        tip = num_nodes - 1
        entries = [4 * 7 + 2, 4 * 8 + 2, 4 * 6 + 2, 4 * 7 + 1, 4 * 3 + 2, 4 * 6 + 1, 4 * 3 + 3, 4 * 6 + 3]
        _initial_velocity_check(
            self,
            build,
            lambda acts: _TipLoss(acts[0], tip, [0.6, 0.1, 0.3]),
            v0,
            num_steps=20,
            dt=0.005,
            entries=entries,
            tol=1e-4,
        )


class RodRigidConstraintAdjointTest(unittest.TestCase):
    """A rod pinned at the top carrying a rigid cube through a node-to-rigid
    constraint: the gradient of the cube's final position w.r.t. the cube's
    initial velocity (rigid accessor) and the rod's initial nodal velocities flows
    through the constraint and the rod's elasticity."""

    def test_cube_velocity_gradient_through_the_rod(self) -> None:
        dt, num_steps = 0.005, 20
        goal = np.array([0.05, 0.0, 0.2])
        v0_cube = np.array([0.3, 0.0, 0.0])

        def build():
            return scenes.rod_with_cube(cube_velocity=tuple(v0_cube))

        scene, rod, cube = build()
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        result = DifferentiableRollout(scene, dt=dt, num_steps=num_steps).run(
            apply_inputs=lambda step: None, terminal_losses=[_CubeLoss(cube, goal)]
        )
        self.assertTrue(result.fd_valid, result.flagged_steps)
        grad = result.gradients["cube"].initial_velocity[:3]

        def rollout_loss(v):
            sc, rd, cb = scenes.rod_with_cube(cube_velocity=tuple(v))
            try:
                configure_for_differentiability(sc)
                loss = _CubeLoss(cb, goal)
                for _ in range(num_steps):
                    sc.step(dt)
                return loss.value()
            finally:
                physics.destroy_scene(sc)

        fd = np.zeros(3)
        for k in range(3):
            fds = []
            for eps in (1e-5, 1e-6):
                dv = np.zeros(3)
                dv[k] = eps
                fds.append((rollout_loss(v0_cube + dv) - rollout_loss(v0_cube - dv)) / (2.0 * eps))
            fd[k] = fds[0]
            if abs(fds[0]) > 1e-8:
                self.assertLessEqual(abs(fds[0] - fds[1]) / abs(fds[0]), 1e-4, "rough FD")
        # Norm-relative: the scene is symmetric in y, so that component's true
        # derivative vanishes (a component-relative error would be meaningless).
        self.assertGreater(np.linalg.norm(fd), 0.0)
        rel = np.linalg.norm(grad - fd) / np.linalg.norm(fd)
        self.assertLessEqual(rel, 1e-5, (grad, fd))

    def test_rod_velocity_gradient_through_the_constraint(self) -> None:
        def build():
            scene, rod, cube = scenes.rod_with_cube()
            return scene, (rod, cube)

        scene, (rod, cube) = build()
        n = rod.get_num_dofs()
        physics.destroy_scene(scene)
        v0 = np.zeros(n)
        v0[0::4] = 0.2
        v0[3::4] = 1.0
        entries = [4 * 3 + 0, 4 * 5 + 0, 4 * 6 + 0, 4 * 4 + 2, 4 * 2 + 1, 4 * 3 + 3]
        _initial_velocity_check(
            self,
            build,
            lambda acts: _CubeLoss(acts[1], [0.05, 0.0, 0.2]),
            v0,
            num_steps=20,
            dt=0.005,
            entries=entries,
            tol=1e-4,
        )


class RodStaticContactAdjointTest(unittest.TestCase):
    """A rod landing on a static ground plane: centerline contact against a static
    collider goes through the generic deformable previous-state contact path
    (the same as soft bodies), with the stage-start SDF Hessian of the plane."""

    def test_initial_velocity_gradients_through_contact(self) -> None:
        def build():
            scene, rod = scenes.rod_on_plane("coulomb")
            return scene, (rod,)

        scene, (rod,) = build()
        n = rod.get_num_dofs()
        num_nodes = n // 4
        physics.destroy_scene(scene)
        v0 = np.zeros(n)
        v0[2::4] = -0.5
        v0[0::4] = 0.3
        # Verify the rollout actually touches the plane.
        sc, rd = scenes.rod_on_plane("coulomb")
        configure_for_differentiability(sc)
        min_z = np.inf
        for _ in range(30):
            sc.step(0.005)
            min_z = min(min_z, min(_node_position(rd, i)[2] for i in range(num_nodes)))
        physics.destroy_scene(sc)
        self.assertLess(min_z, 0.005, "test is vacuous: the rod never reached the plane")
        entries = [4 * 3 + 2, 4 * 6 + 2, 4 * 0 + 0, 4 * 3 + 0, 4 * 6 + 0, 4 * 4 + 1, 4 * 2 + 3]
        _initial_velocity_check(
            self,
            build,
            lambda acts: _TipLoss(acts[0], num_nodes - 1, [0.4, 0.0, 0.05]),
            v0,
            num_steps=30,
            dt=0.005,
            entries=entries,
            tol=1e-4,
        )


class RodArticulatedIslandTest(unittest.TestCase):
    """A stiff rod tied to a controlled pendulum's link (:func:`scenes.rod_on_pendulum`):
    one island with an articulated actor, its pose controller and a rod.

    Pins two engine defects found on the tendon-driven finger (2026-09-02):

    * the controller's input-target assembly resized the rod's residual to its (empty)
      input rows and the parameter-gradient pass then asserted "Residual must not be
      empty" (heap corruption without assertions) - any island mixing a rod with a
      controlled articulated actor crashed in back_propagate;
    * the adjoint solve stopped on an absolute residual tolerance (1e-3 by default),
      i.e. on a scale set by the loss: on this scene the production defaults gave a
      controller gradient 2 percent off, and the same loss scaled by 1e-4 an exactly
      zero gradient with the finite-difference self-check passing. The criterion is
      now relative to |rhs|."""

    dt, num_steps = 0.005, 16
    goal = np.array([0.25, 0.02, 0.76])

    @classmethod
    def targets(cls):
        n = cls.num_steps
        return np.stack([np.linspace(0.0, 0.3, n), np.linspace(0.0, -0.2, n)], axis=1)

    class _WeightedTipLoss(_TipLoss):
        def __init__(self, rod, node, goal, weight):
            super().__init__(rod, node, goal)
            self.weight = weight

        def value(self) -> float:
            return self.weight * super().value()

        def accumulate_output_grad(self) -> None:
            d = _node_position(self.rod, self.node) - self.goal
            g = np.zeros(self.rod.get_num_dofs())
            g[4 * self.node : 4 * self.node + 3] = self.weight * d
            diffsim.get_displacements_backward(self.rod, g)

    def _control_gradient(self, weight: float, default_solver: bool):
        """Adjoint dL/d(targets) (2 x num_steps) of the weighted node-3 loss; with
        ``default_solver`` the engine's default adjoint solver parameters (plus the
        finite-difference validation) replace the harness's tight ones."""
        scene, chain, rod = scenes.rod_on_pendulum()
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        if default_solver:
            dp = diffsim.BackPropagationSolverParams()
            dp.validate_finite_diff = True
            diffsim.set_back_propagation_solver_params(scene, dp)
        targets = self.targets()
        result = DifferentiableRollout(scene, dt=self.dt, num_steps=self.num_steps).run(
            apply_inputs=lambda step: chain.set_articulated_target_pose(
                np.ascontiguousarray(targets[step])
            ),
            terminal_losses=[self._WeightedTipLoss(rod, 3, self.goal, weight)],
        )
        self.assertTrue(result.fd_valid, result.flagged_steps)
        self.assertEqual(result.minres_fallbacks, 0)
        return result, result.gradients[chain.get_name()].control_targets.copy()

    def _fd_control_gradient(self, dof: int, step: int, eps: float) -> float:
        targets = self.targets()

        def loss_at(c):
            sc, ch, rd = scenes.rod_on_pendulum()
            try:
                configure_for_differentiability(sc)
                loss = _TipLoss(rd, 3, self.goal)
                for s in range(self.num_steps):
                    ch.set_articulated_target_pose(np.ascontiguousarray(c[s]))
                    sc.step(self.dt)
                return loss.value()
            finally:
                physics.destroy_scene(sc)

        plus, minus = targets.copy(), targets.copy()
        plus[step, dof] += eps
        minus[step, dof] -= eps
        return (loss_at(plus) - loss_at(minus)) / (2.0 * eps)

    def test_control_gradients_match_finite_differences(self) -> None:
        result, grad = self._control_gradient(1.0, default_solver=False)
        self.assertLessEqual(result.max_adjoint_residual, 1e-6)
        rel_errors, skipped = {}, {}
        for dof, step in ((0, 2), (1, 2), (0, 8), (1, 8)):
            fds = [self._fd_control_gradient(dof, step, eps) for eps in (1e-5, 1e-6)]
            denom = max(abs(fds[0]), 1e-30)
            fd_self = abs(fds[0] - fds[1]) / denom
            if fd_self > 1e-4:
                skipped[(dof, step)] = fd_self
                continue
            rel_errors[(dof, step)] = abs(grad[dof, step] - fds[0]) / denom
        self.assertGreaterEqual(len(rel_errors), 2, f"too few smooth entries: {skipped}")
        self.assertLessEqual(max(rel_errors.values()), 1e-4, (rel_errors, skipped))

    def test_default_solver_gradients_are_scale_invariant(self) -> None:
        _, tight = self._control_gradient(1.0, default_solver=False)
        _, default = self._control_gradient(1.0, default_solver=True)
        _, default_small = self._control_gradient(1e-4, default_solver=True)
        scale = np.linalg.norm(tight)
        self.assertGreater(scale, 0.0)
        # Former absolute default (1e-3): 1.9e-2 here.
        self.assertLessEqual(np.linalg.norm(default - tight) / scale, 1e-5)
        # Former absolute default: exactly zero gradient (relative error 1).
        self.assertLessEqual(np.linalg.norm(default_small / 1e-4 - tight) / scale, 1e-5)


class RodDynamicContactAdjointTest(unittest.TestCase):
    """Sync contact of the rod's centerline samples against a dynamic box sliding on
    the ground (:func:`scenes.rod_onto_cube`, one island): gradients of the rod's
    tip position w.r.t. the cube's initial velocity (rigid accessor) and the rod's
    initial nodal velocities, through the frictional rod-cube contact."""

    dt, num_steps = 0.005, 20
    goal = np.array([0.3, 0.0, 0.1])

    def test_initial_velocity_gradients_through_dynamic_contact(self) -> None:
        v0_cube = np.array([0.3, 0.0, 0.0])
        scene, rod, cube = scenes.rod_onto_cube(cube_velocity=tuple(v0_cube))
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        v0_rod = np.zeros(rod.get_num_dofs())
        v0_rod[2::4] = -0.5  # the scene's initial rod velocities (no getter exists)
        result = DifferentiableRollout(scene, dt=self.dt, num_steps=self.num_steps).run(
            apply_inputs=lambda step: None, terminal_losses=[_TipLoss(rod, 6, self.goal)]
        )
        self.assertTrue(result.fd_valid, result.flagged_steps)
        self.assertEqual(result.minres_fallbacks, 0)
        grad_cube = result.gradients["cube"].initial_velocity[:3]
        grad_rod = result.gradients["rod"].initial_velocity

        def rollout(v_cube, v_rod):
            sc, rd, cb = scenes.rod_onto_cube(cube_velocity=tuple(v_cube))
            try:
                configure_for_differentiability(sc)
                rd.set_node_velocities_local(v_rod)
                loss = _TipLoss(rd, 6, self.goal)
                for _ in range(self.num_steps):
                    sc.step(self.dt)
                return loss.value(), _node_position(rd, 3)[2]
            finally:
                physics.destroy_scene(sc)

        def rollout_loss(v_cube, v_rod):
            return rollout(v_cube, v_rod)[0]

        # The rod must have reached the cube's top face (else the island never couples).
        self.assertLess(rollout(v0_cube, v0_rod)[1], 0.205)

        fd_cube = np.zeros(3)
        for k in range(3):
            fds = []
            for eps in (1e-5, 1e-6):
                dv = np.zeros(3)
                dv[k] = eps
                fds.append(
                    (rollout_loss(v0_cube + dv, v0_rod) - rollout_loss(v0_cube - dv, v0_rod))
                    / (2.0 * eps)
                )
            fd_cube[k] = fds[0]
            if abs(fds[0]) > 1e-8:
                self.assertLessEqual(abs(fds[0] - fds[1]) / abs(fds[0]), 1e-4, "rough FD")
        self.assertGreater(np.linalg.norm(fd_cube), 0.0)
        rel = np.linalg.norm(grad_cube - fd_cube) / np.linalg.norm(fd_cube)
        self.assertLessEqual(rel, 1e-4, (grad_cube, fd_cube))

        # Rod initial velocities. The directional derivative along the gradient is the
        # clean check (single entries of the fall velocity move the contact onset and
        # their rollout FD is self-consistent to 2e-4..3e-4 only); the per-entry rows
        # are checked at their own FD self-consistency level.
        direction = grad_rod / np.linalg.norm(grad_rod)
        fds = []
        for eps in (1e-5, 1e-6):
            fds.append(
                (
                    rollout_loss(v0_cube, v0_rod + eps * direction)
                    - rollout_loss(v0_cube, v0_rod - eps * direction)
                )
                / (2.0 * eps)
            )
        self.assertLessEqual(abs(fds[0] - fds[1]) / abs(fds[0]), 1e-4, "rough directional FD")
        self.assertLessEqual(abs(np.linalg.norm(grad_rod) - fds[0]) / abs(fds[0]), 1e-4, fds)
        rel_errors, skipped = {}, {}
        for k in (4 * 2 + 2, 4 * 3 + 2, 4 * 4 + 2, 4 * 3 + 0):
            fds = []
            for eps in (1e-5, 1e-6):
                dv = np.zeros_like(v0_rod)
                dv[k] = eps
                fds.append(
                    (rollout_loss(v0_cube, v0_rod + dv) - rollout_loss(v0_cube, v0_rod - dv))
                    / (2.0 * eps)
                )
            denom = max(abs(fds[0]), 1e-30)
            fd_self = abs(fds[0] - fds[1]) / denom
            if fd_self > 1e-3:
                skipped[k] = fd_self
                continue
            rel_errors[k] = abs(grad_rod[k] - fds[0]) / denom
        self.assertGreaterEqual(len(rel_errors), 3, f"too few smooth entries: {skipped}")
        self.assertLessEqual(max(rel_errors.values()), 1e-3, (rel_errors, skipped))


class RodSupportBoundaryTest(unittest.TestCase):
    """The unsupported cases must fail loudly, never silently drop a term."""

    def test_stiffness_damping_is_rejected(self) -> None:
        ex = physics.experimental
        scene = physics.create_scene("rod_damping")
        self.addCleanup(physics.destroy_scene, scene)
        material = ex.RodMaterialParams(stiffness_damping_coefficient=0.1)
        nodes = np.stack([np.linspace(0, 0.3, 5), np.zeros(5), np.full(5, 0.5)], axis=1)
        scenes._rod_actor(scene, nodes, material=material)
        with self.assertRaisesRegex(Exception, "stiffness damping"):
            diffsim.make_scene_differentiable(scene)

    def _rod_onto_cube_backward(self, **kwargs) -> None:
        scene, rod, cube = scenes.rod_onto_cube(**kwargs)
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        DifferentiableRollout(scene, dt=0.005, num_steps=20).run(
            apply_inputs=lambda step: None,
            terminal_losses=[_TipLoss(rod, 6, [0.3, 0.0, 0.1])],
        )

    def test_rod_as_collider_of_a_dynamic_actor_is_rejected(self) -> None:
        # The cube's samples against the rod's point-cloud collider: no stage-start
        # SDF Hessian and no stage-start collider Jacobian on the rod side.
        with self.assertRaisesRegex(Exception, "collider"):
            self._rod_onto_cube_backward(rod_as_collider=True)

    def test_mesh_collider_cube_velocity_gradient(self) -> None:
        """The rod's samples against the cube's triangle-mesh collider (closest-point queries,
        the signed distance's Hessian by the closest feature since 2026-09-05; mesh colliders
        were refused before): the gradient of the rod's tip position with respect to the
        cube's initial velocity against central FD, as in RodDynamicContactAdjointTest.
        Measured 1.2e-4 relative (x 1.1e-4, y 1.4e-5, z 7e-4 on a component ten times
        smaller), against 3e-5 with the box collider of the same cube (x 2.4e-5, z 1.7e-4).
        These are not adjoint errors: the loss is piecewise smooth (the rod's contact set
        switches), and its difference quotients spread by 3e-3 (x, y) to 4e-2 (z) across
        eps 1e-6..1e-9, a band the adjoint lies within (2026-09-06). A rigid cube on a static
        mesh box or the chain pushing a mesh-collider cube are exact to 5e-7 / 1e-4
        (test_diffsim_gradients)."""
        dt, num_steps, goal = 0.005, 20, np.array([0.3, 0.0, 0.1])
        v0_cube = np.array([0.3, 0.0, 0.0])
        scene, rod, cube = scenes.rod_onto_cube(
            cube_velocity=tuple(v0_cube), cube_collider=physics.ColliderType.MESH
        )
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        result = DifferentiableRollout(scene, dt=dt, num_steps=num_steps).run(
            apply_inputs=lambda step: None, terminal_losses=[_TipLoss(rod, 6, goal)]
        )
        self.assertTrue(result.fd_valid, result.flagged_steps)
        grad_cube = result.gradients["cube"].initial_velocity[:3]

        def rollout_loss(v_cube):
            sc, rd, cb = scenes.rod_onto_cube(
                cube_velocity=tuple(v_cube), cube_collider=physics.ColliderType.MESH
            )
            try:
                configure_for_differentiability(sc)
                loss = _TipLoss(rd, 6, goal)
                for _ in range(num_steps):
                    sc.step(dt)
                self.assertLess(_node_position(rd, 3)[2], 0.205, "the rod must reach the cube")
                return loss.value()
            finally:
                physics.destroy_scene(sc)

        fd_cube = np.zeros(3)
        for k in range(3):
            fds = []
            for eps in (1e-5, 1e-6):
                dv = np.zeros(3)
                dv[k] = eps
                fds.append((rollout_loss(v0_cube + dv) - rollout_loss(v0_cube - dv)) / (2.0 * eps))
            fd_cube[k] = fds[0]
            if abs(fds[0]) > 1e-8:
                self.assertLessEqual(abs(fds[0] - fds[1]) / abs(fds[0]), 1e-4, "rough FD")
        self.assertGreater(np.linalg.norm(fd_cube), 0.0)
        rel = np.linalg.norm(grad_cube - fd_cube) / np.linalg.norm(fd_cube)
        self.assertLessEqual(rel, 3e-4, (grad_cube, fd_cube))
        self.assertLessEqual(abs(grad_cube[0] - fd_cube[0]) / abs(fd_cube[0]), 2e-4, (grad_cube, fd_cube))


if __name__ == "__main__":
    unittest.main()
