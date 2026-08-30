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

"""Validation of the parameter gradients: ``diffsim.set_gravity_backward`` and
``diffsim.set_contact_params_backward``.

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


CONTACT_PARAM_FIELDS = (
    "penalty_coefficient",
    "coulomb_friction_coefficient",
    "viscous_friction_coefficient",
    "normal_viscous_damping_coefficient",
)


def adjoint_contact_grads(scene, losses, actors) -> dict:
    """Sweep the adjoint and read each actor's contact-parameter gradient."""
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
    out = {}
    for name, actor in actors.items():
        grad = np.zeros(len(CONTACT_PARAM_FIELDS))
        diffsim.set_contact_params_backward(actor, grad)
        out[name] = grad
    for handle in pre + post:
        scene.release_state(handle)
    return out


def fd_contact_grads(scene, losses, actor, state_init) -> np.ndarray:
    """Independent ground truth: central differences over set_contact_params."""
    fd = np.zeros(len(CONTACT_PARAM_FIELDS))
    for f_i, field in enumerate(CONTACT_PARAM_FIELDS):
        base = getattr(actor.get_contact_params(), field)
        eps = FD_EPS * (1.0 + abs(base))
        values = []
        for sign in (+1.0, -1.0):
            scene.restore_state(state_init, False)
            perturbed = actor.get_contact_params()
            setattr(perturbed, field, base + sign * eps)
            actor.set_contact_params(perturbed)
            for _ in range(NUM_STEPS):
                scene.step(DT)
            values.append(sum(loss.value() for loss in losses))
            restored = actor.get_contact_params()
            setattr(restored, field, base)
            actor.set_contact_params(restored)
        fd[f_i] = (values[0] - values[1]) / (2.0 * eps)
    return fd


def _rich_contact_scene():
    """Cube sliding on a plane with every differentiated parameter active."""
    scene = physics.create_scene("contact_params_rich")
    scene.set_gravity([0.0, 0.0, -9.81])
    cp = physics.ContactParams(
        penalty_coefficient=1e8,
        coulomb_friction_coefficient=0.4,
        viscous_friction_coefficient=0.1,
        normal_viscous_damping_coefficient=5.0,
    )
    ground = scene.create_rigid_actor(
        name="ground",
        shape=physics.create_plane_shape(normal=[0, 0, 1], distance=0.0),
        is_static=True,
        contact=cp,
    )
    cube = scene.create_rigid_actor(
        name="cube",
        shape=physics.create_tet_mesh_shape(
            coordinates=scenes.CUBE_COORDS, connectivity=scenes.CUBE_CONN
        ),
        density=1000.0,
        contact=cp,
        world_from_local=physics.TransformRT([0.0, 0.0, 0.099]),
    )
    cube.set_velocity([0.5, 0.0, 0.0], [0.0, 0.0, 0.0])
    return scene, cube, ground


class ContactParamsGradientTest(unittest.TestCase):
    def _compare_per_field(self, adjoint, fd, tol) -> None:
        """Per-field comparison so the hugely different scales (a penalty
        gradient of ~1e-11 next to a friction gradient of ~1e-2) each get
        checked instead of the small one hiding inside a vector norm."""
        floor = 1e-13
        for f_i, field in enumerate(CONTACT_PARAM_FIELDS):
            denom = max(abs(adjoint[f_i]), abs(fd[f_i]))
            if denom <= floor:
                continue  # both zero at working precision - consistent
            rel = abs(adjoint[f_i] - fd[f_i]) / denom
            self.assertLessEqual(
                rel,
                tol,
                f"{field}: adjoint={adjoint[f_i]:.6e}, fd={fd[f_i]:.6e}",
            )

    def test_rigid_on_plane_all_fields_vs_fd(self) -> None:
        scene, cube, _ground = _rich_contact_scene()
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        loss = TranslationErrorLoss(cube)
        state_init = scene.capture_state()
        grads = adjoint_contact_grads(scene, [loss], {"cube": cube})
        fd = fd_contact_grads(scene, [loss], cube, state_init)
        scene.release_all_states()
        self.assertTrue(np.any(np.abs(fd) > 0.0), "test is vacuous")
        self._compare_per_field(grads["cube"], fd, 1e-2)

    def test_static_collider_read_is_an_error(self) -> None:
        """Static colliders are not island members; reading must fail loudly,
        never return a silent zero."""
        scene, cube, ground = _rich_contact_scene()
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        adjoint_contact_grads(scene, [TranslationErrorLoss(cube)], {"cube": cube})
        with self.assertRaises(physics.Error):
            out = np.zeros(len(CONTACT_PARAM_FIELDS))
            diffsim.set_contact_params_backward(ground, out)
        scene.release_all_states()

    def test_no_contact_gives_exact_zeros(self) -> None:
        """An actor in a swept island with no contacts has an exactly zero
        gradient (the component exists - distinct from the never-swept error)."""
        scene, cube = scenes.rigid_free()
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        grads = adjoint_contact_grads(
            scene, [TranslationErrorLoss(cube)], {"cube": cube}
        )
        scene.release_all_states()
        np.testing.assert_array_equal(
            grads["cube"], np.zeros(len(CONTACT_PARAM_FIELDS))
        )

    def test_read_before_any_sweep_is_an_error(self) -> None:
        scene, cube = scenes.rigid_free()
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        diffsim.reset_back_propagation(scene)
        with self.assertRaises(physics.Error):
            out = np.zeros(len(CONTACT_PARAM_FIELDS))
            diffsim.set_contact_params_backward(cube, out)

    def test_articulated_link_params_vs_fd(self) -> None:
        """Contact parameters live on nested link actors for articulations."""
        # "rich" params: all differentiated fields strictly positive so the
        # central FD ground truth never crosses the non-negativity validation.
        # root_z=0.105: the links start essentially on the plane so contact
        # (and therefore a nonzero parameter gradient) exists from step one.
        scene, chain = scenes.free_chain_on_plane("rich", root_z=0.105)
        # Tilted gravity makes the chain slide, so tangential velocity (and
        # with it the friction-coefficient gradient) is genuinely nonzero
        # instead of sitting at the epsilon noise floor of a vertical drop.
        scene.set_gravity([2.0, 0.0, -9.81])
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        links = []
        scene.for_each_actor(
            lambda a: links.append(a) if a.is_nested_link_actor() else None
        )
        self.assertGreaterEqual(len(links), 2)
        link = links[0]
        ref = np.zeros(chain.get_num_dofs())
        ref[-1] = 0.3
        loss = ArticulatedPoseErrorLoss(chain, ref)
        state_init = scene.capture_state()
        grads = adjoint_contact_grads(scene, [loss], {"link": link})
        fd = fd_contact_grads(scene, [loss], link, state_init)
        scene.release_all_states()
        self.assertTrue(np.any(np.abs(fd) > 0.0), "test is vacuous")
        self._compare_per_field(grads["link"], fd, 3e-2)

    def test_reset_zeroes_contact_grads(self) -> None:
        scene, cube, _ground = _rich_contact_scene()
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        grads = adjoint_contact_grads(
            scene, [TranslationErrorLoss(cube)], {"cube": cube}
        )
        self.assertTrue(np.any(np.abs(grads["cube"]) > 0.0))
        diffsim.reset_back_propagation(scene)
        out = np.ones(len(CONTACT_PARAM_FIELDS))
        diffsim.set_contact_params_backward(cube, out)
        np.testing.assert_array_equal(out, np.zeros(len(CONTACT_PARAM_FIELDS)))
        scene.release_all_states()


def _sweep(scene, losses, apply_inputs=None):
    """Forward rollout + full reverse sweep (terminal losses only)."""
    pre, post = [], []
    for step in range(NUM_STEPS):
        if apply_inputs is not None:
            apply_inputs(step)
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
    for handle in pre + post:
        scene.release_state(handle)


def adjoint_density_grad(scene, losses, actor, apply_inputs=None) -> float:
    _sweep(scene, losses, apply_inputs)
    grad = np.zeros(1)
    diffsim.set_density_backward(actor, grad)
    return float(grad[0])


def fd_density_grad(scene, losses, actor, state_init, apply_inputs=None) -> float:
    density0 = actor.get_density()
    eps = 1e-4 * density0  # relative step; density is strictly positive
    values = []
    for sign in (+1.0, -1.0):
        scene.restore_state(state_init, False)
        actor.set_density(density0 + sign * eps)
        for step in range(NUM_STEPS):
            if apply_inputs is not None:
                apply_inputs(step)
            scene.step(DT)
        values.append(sum(loss.value() for loss in losses))
        actor.set_density(density0)
    return (values[0] - values[1]) / (2.0 * eps)


class DensityGradientTest(unittest.TestCase):
    def _free_cube(self, gravity):
        scene = physics.create_scene("density_test")
        scene.set_gravity(gravity)
        cube = scene.create_rigid_actor(
            name="cube",
            shape=physics.create_tet_mesh_shape(
                coordinates=scenes.CUBE_COORDS, connectivity=scenes.CUBE_CONN
            ),
            density=1000.0,
            world_from_local=physics.TransformRT([0.0, 0.0, 1.0]),
        )
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        return scene, cube

    def test_free_fall_mass_invariance(self) -> None:
        """Under gravity alone every residual term is proportional to the
        density, so at a converged step dR/drho = R/rho ~ (solver residual)/rho
        and the density gradient must vanish - a sharp analytic invariance, not
        a tolerance tuned to pass. The same sweep's gravity gradient is checked
        to be nonzero so the test cannot pass vacuously."""
        scene, cube = self._free_cube([0.0, 0.0, -9.81])
        cube.set_velocity([0.3, -0.2, 0.1], [0.0, 0.0, 0.0])
        loss = TranslationErrorLoss(cube)
        state_init = scene.capture_state()
        grad = adjoint_density_grad(scene, [loss], cube)
        gravity_grad = np.zeros(3)
        diffsim.set_gravity_backward(scene, gravity_grad)
        self.assertGreater(np.linalg.norm(gravity_grad), 1e-4)
        self.assertLessEqual(abs(grad), 1e-8, f"free-fall density gradient {grad}")
        fd = fd_density_grad(scene, [loss], cube, state_init)
        scene.release_all_states()
        self.assertLessEqual(abs(fd), 1e-8, f"free-fall FD density gradient {fd}")

    def test_external_force_closed_form_and_fd(self) -> None:
        """Zero gravity, constant per-step force F: a = F/m, so under Backward
        Euler p_N - p_0 = dt^2 N(N+1)/2 F/m and

            dL/drho = -(dt^2 N(N+1)/2) F^T (p_N - ref) / (m rho).
        """
        scene, cube = self._free_cube([0.0, 0.0, 0.0])
        force_dofs = np.array([0, 1, 2], dtype=np.int32)
        force = np.array([2.0, 0.0, 0.0])

        def apply_inputs(_step: int) -> None:
            cube.set_external_forces_on_dofs(force_dofs, force)

        loss = TranslationErrorLoss(cube)
        state_init = scene.capture_state()
        grad = adjoint_density_grad(scene, [loss], cube, apply_inputs)

        scene.restore_state(state_init, False)
        for step in range(NUM_STEPS):
            apply_inputs(step)
            scene.step(DT)
        p_final = np.asarray(
            cube.get_center_of_mass_transform().translation, dtype=np.float64
        )
        mass = cube.get_mass()
        density = cube.get_density()
        closed_form = (
            -(DT * DT * NUM_STEPS * (NUM_STEPS + 1) / 2.0)
            * float(force @ (p_final - loss.ref))
            / (mass * density)
        )
        rel = abs(grad - closed_form) / abs(closed_form)
        self.assertLessEqual(
            rel, 1e-5, f"closed-form mismatch: adjoint={grad}, closed={closed_form}"
        )
        fd = fd_density_grad(scene, [loss], cube, state_init, apply_inputs)
        scene.release_all_states()
        self.assertLessEqual(
            abs(grad - fd) / abs(fd), 1e-4, f"FD mismatch: adjoint={grad}, fd={fd}"
        )

    def test_rigid_contact_vs_fd(self) -> None:
        scene, cube = scenes.rigid_on_plane("rich")
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        loss = TranslationErrorLoss(cube)
        state_init = scene.capture_state()
        grad = adjoint_density_grad(scene, [loss], cube)
        fd = fd_density_grad(scene, [loss], cube, state_init)
        scene.release_all_states()
        self.assertGreater(abs(fd), 0.0, "test is vacuous")
        self.assertLessEqual(
            abs(grad - fd) / max(abs(grad), abs(fd)),
            1e-2,
            f"adjoint={grad}, fd={fd}",
        )

    def test_articulated_link_vs_fd(self) -> None:
        scene, chain = scenes.free_chain_on_plane("rich", root_z=0.105)
        scene.set_gravity([2.0, 0.0, -9.81])
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        links = []
        scene.for_each_actor(
            lambda a: links.append(a) if a.is_nested_link_actor() else None
        )
        link = links[0]
        ref = np.zeros(chain.get_num_dofs())
        ref[-1] = 0.3
        loss = ArticulatedPoseErrorLoss(chain, ref)
        state_init = scene.capture_state()
        grad = adjoint_density_grad(scene, [loss], link)
        fd = fd_density_grad(scene, [loss], link, state_init)
        scene.release_all_states()
        self.assertGreater(abs(fd), 0.0, "test is vacuous")
        self.assertLessEqual(
            abs(grad - fd) / max(abs(grad), abs(fd)),
            1e-2,
            f"adjoint={grad}, fd={fd}",
        )

    def test_compound_actor_has_no_inertia_error(self) -> None:
        """The articulated compound itself carries no rigid-body inertia; the
        inertia (and so the density gradient) lives on its links."""
        scene, chain = scenes.free_chain_on_plane("rich", root_z=0.105)
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        ref = np.zeros(chain.get_num_dofs())
        _sweep(scene, [ArticulatedPoseErrorLoss(chain, ref)])
        with self.assertRaises(physics.Error):
            out = np.zeros(1)
            diffsim.set_density_backward(chain, out)
        scene.release_all_states()

    def test_reset_and_never_swept_contracts(self) -> None:
        scene, cube = scenes.rigid_on_plane("rich")
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        diffsim.reset_back_propagation(scene)
        with self.assertRaises(physics.Error):
            out = np.zeros(1)
            diffsim.set_density_backward(cube, out)
        grad = adjoint_density_grad(scene, [TranslationErrorLoss(cube)], cube)
        self.assertNotEqual(grad, 0.0)
        diffsim.reset_back_propagation(scene)
        out = np.ones(1)
        diffsim.set_density_backward(cube, out)
        self.assertEqual(out[0], 0.0)
        scene.release_all_states()


if __name__ == "__main__":
    unittest.main()
