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

"""Tests for :mod:`superdex.physics.diffsim_torch`.

The decisive check is ``torch.autograd.gradcheck``: torch numerically
differentiates the bridge (central differences on every input entry) and
compares against the engine adjoint the bridge returns - an independent
validation path on top of the suite's own FD tests.

- gradcheck over controller targets on the pendulum (articulated path);
- gradcheck over linear external forces + gravity + contact parameters +
  density on a cube-on-plane contact scene, with the physical parameters
  reached through a smooth reparameterization so every gradcheck perturbation
  is well-scaled (this also exercises chaining into an upstream torch graph),
  torques included (:class:`TorqueGradientTest` measures the torque gradients
  against finite differences on their own);
- bridge loss equals the direct DifferentiableRollout loss;
- repeated calls are deterministic (bitwise-equal loss and gradients);
- gradcheck over a soft cube's initial nodal state + gravity + contact
  parameters on the soft-on-plane contact scene (the ``initial_states``
  group), and over the soft material parameters (E, nu, rho, mass damping;
  the ``soft_materials`` group) through a reparameterization;
- contract violations (dtype, device via meta, shape, undeclared/missing
  groups, static contact actor, controller-less control actor, soft actors
  in the force/density groups, a rigid actor in the initial-state group)
  raise loudly.

Requires SUPERDEX_PRECISION=double; skips (loudly) if torch is unavailable.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np
import superdex.physics as physics

diffsim = physics.diffsim

try:
    import torch
except ImportError:
    torch = None

from . import scenes
from .harness import (
    ArticulatedPoseErrorLoss,
    DisplacementErrorLoss,
    QuaternionErrorLoss,
    TranslationErrorLoss,
    configure_for_differentiability,
    real_dtype,
)

_NUM_WORKER_THREADS = int(os.environ.get("SUPERDEX_DIFFSIM_TEST_THREADS", "0"))
DT = 0.01


def _assert_step_converged(test: unittest.TestCase, scene, step: int) -> None:
    """The forward step just taken solved its dynamics: residual below 1e-6 N. A converged
    step ends at a residual of 1e-12..1e-9 N, whether the solver reports CONVERGED or STOPPED
    (STOPPED is a line search that cannot reduce a residual already at rounding level); a
    step that hit the iteration cap in a diverging rollout ends at residuals of 1e-3..10 N."""
    stats = scene.get_solver_stats()
    test.assertLess(stats.residual_norm, 1e-6, f"step {step}: {stats.convergence_status}, residual {stats.residual_norm:.2e}")


def setUpModule() -> None:
    if torch is None:
        raise unittest.SkipTest(
            "torch is not installed; install PyTorch to run the bridge tests"
        )
    if not physics.uses_double_precision():
        raise unittest.SkipTest("torch bridge tests require SUPERDEX_PRECISION=double")
    physics.initialize(num_worker_threads=_NUM_WORKER_THREADS)


def tearDownModule() -> None:
    if physics.is_initialized():
        physics.shutdown()


def _make_bridge_module():
    # Imported lazily: the module import itself requires torch.
    from superdex.physics import diffsim_torch

    return diffsim_torch


class GradcheckControlsTest(unittest.TestCase):
    def test_pendulum_controller_targets(self) -> None:
        diffsim_torch = _make_bridge_module()
        num_steps = 3
        scene, chain = scenes.pendulum(with_controller=True)
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)

        bridge = diffsim_torch.TorchRollout(
            scene,
            dt=DT,
            num_steps=num_steps,
            control_actors=[chain],
            terminal_losses=[
                ArticulatedPoseErrorLoss(chain, np.array([0.4, -0.2]))
            ],
        )
        self.addCleanup(bridge.close)
        self.assertEqual(bridge.control_size, 2)

        controls = torch.zeros(
            (num_steps, 2), dtype=torch.float64, requires_grad=True
        )
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda c: bridge(controls=c),
                (controls,),
                eps=1e-6,
                atol=1e-8,
                rtol=1e-4,
            )
        )
        self.assertTrue(bridge.last_result.fd_valid)

    def test_control_actor_without_controller_is_an_error(self) -> None:
        diffsim_torch = _make_bridge_module()
        scene, chain = scenes.pendulum(with_controller=False)
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        with self.assertRaisesRegex(ValueError, "controller"):
            diffsim_torch.TorchRollout(
                scene,
                dt=DT,
                num_steps=2,
                control_actors=[chain],
                terminal_losses=[
                    ArticulatedPoseErrorLoss(chain, np.array([0.4, -0.2]))
                ],
            )


class GradcheckParametersTest(unittest.TestCase):
    """Forces + gravity + contact material + density through one graph."""

    def _build(self):
        diffsim_torch = _make_bridge_module()
        num_steps = 3
        scene, cube = scenes.rigid_on_plane("rich")
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        bridge = diffsim_torch.TorchRollout(
            scene,
            dt=DT,
            num_steps=num_steps,
            force_actors=[cube],
            contact_actors=[cube],
            density_actors=[cube],
            differentiate_gravity=True,
            terminal_losses=[TranslationErrorLoss(cube)],
        )
        self.addCleanup(bridge.close)
        return diffsim_torch, bridge, num_steps

    def test_gradcheck_all_parameter_groups(self) -> None:
        _, bridge, num_steps = self._build()
        self.assertEqual(bridge.force_size, 6)

        # The cube's "rich" material; perturbing these hugely different
        # scales directly would be numerically meaningless for gradcheck, so
        # the raw parameters are reached through a smooth, well-scaled
        # reparameterization (which also validates chaining into an upstream
        # torch graph).
        contact_base = torch.tensor(
            [[1e8, 0.4, 0.1, 5.0]], dtype=torch.float64
        )
        density_base = torch.tensor([1000.0], dtype=torch.float64)
        # All six force DoFs are gradchecked, torques included: since the adjoint
        # operator carries the moving-chart term of the external torques (see
        # TorqueGradientTest below) the torque gradients are exact to the finite-difference
        # noise of the adjoint's products.

        def f(contact_scale, density_scale, forces, gravity):
            return bridge(
                forces=forces,
                gravity=gravity,
                contact_params=contact_base * (1.0 + 0.05 * contact_scale),
                densities=density_base * (1.0 + 0.05 * density_scale),
            )

        contact_scale = torch.zeros((1, 4), dtype=torch.float64, requires_grad=True)
        density_scale = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        forces = torch.tensor(
            np.concatenate(
                [
                    0.5 * np.sin(np.arange(3 * num_steps, dtype=np.float64)).reshape(num_steps, 3),
                    0.3 * np.cos(np.arange(3 * num_steps, dtype=np.float64)).reshape(num_steps, 3),
                ],
                axis=1,
            ),
            dtype=torch.float64,
            requires_grad=True,
        )
        gravity = torch.tensor(
            [0.0, 0.0, -9.81], dtype=torch.float64, requires_grad=True
        )
        self.assertTrue(
            torch.autograd.gradcheck(
                f,
                (contact_scale, density_scale, forces, gravity),
                eps=1e-6,
                atol=1e-8,
                rtol=1e-4,
            )
        )

    def test_loss_matches_direct_rollout_and_is_deterministic(self) -> None:
        _, bridge, num_steps = self._build()
        forces = torch.full((num_steps, 6), 0.25, dtype=torch.float64)
        gravity = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64)
        contact = torch.tensor([[1e8, 0.4, 0.1, 5.0]], dtype=torch.float64)
        density = torch.tensor([1000.0], dtype=torch.float64)

        def call():
            inputs = dict(
                forces=forces.clone().requires_grad_(True),
                gravity=gravity.clone().requires_grad_(True),
                contact_params=contact.clone().requires_grad_(True),
                densities=density.clone().requires_grad_(True),
            )
            loss = bridge(**inputs)
            loss.backward()
            return loss.item(), {k: v.grad.numpy().copy() for k, v in inputs.items()}

        loss_a, grads_a = call()
        loss_b, grads_b = call()
        self.assertEqual(loss_a, loss_b)
        for key in grads_a:
            np.testing.assert_array_equal(grads_a[key], grads_b[key])
            self.assertTrue(np.all(np.isfinite(grads_a[key])))
        self.assertGreater(
            sum(float(np.abs(g).sum()) for g in grads_a.values()),
            0.0,
            "test is vacuous: every gradient is zero",
        )

        # The bridge must report the same loss as the plain driver on the
        # same scene state and inputs.
        from superdex.physics.diffsim_rollout import DifferentiableRollout

        scene = bridge.scene
        scene.restore_state(bridge._state_init, False)
        cube = bridge._contact_actors[0]
        params = cube.get_contact_params()
        forces_np = forces.numpy()
        dofs = np.arange(6, dtype=np.int32)
        rollout = DifferentiableRollout(scene, dt=DT, num_steps=num_steps)
        result = rollout.run(
            apply_inputs=lambda step: cube.set_external_forces_on_dofs(
                dofs, np.ascontiguousarray(forces_np[step])
            ),
            terminal_losses=[TranslationErrorLoss(cube)],
        )
        self.assertAlmostEqual(result.loss, loss_a, places=12)


class TorqueGradientTest(unittest.TestCase):
    """Rotational gradients under external torques against central finite differences.

    The step residual is evaluated in the chart of the iterate itself (a left rotation
    increment at the current rotation), so the Jacobian of the step map is the derivative of
    the moving-chart residual: the fixed-chart Hessian H the adjoint's finite-difference
    products produce, minus 1/2 [tau]x on the rotation block of every standalone rigid actor
    with an external torque tau (the chart's derivative acting on the state-independent part
    of the residual; ``exp(eta) exp(delta) = exp(delta + eta - delta x eta / 2 + ...)``). Until
    2026-09-05 the adjoint solved with H alone, and the torque and rotational gradients were
    off by half the rotation the torque induces in a step: 2.2e-4 relative at 0.3 N m and
    8.6e-4 at 1.2 N m on a free cube (before 2026-09-03 twice that, when the constant torque
    term was still transported inside the products - the other half of the same term, with
    the wrong sign). With the term in the operator (both the Krylov and the Newton outer
    solver) the gradients are exact to the finite differences' resolution: 6.5e-10 at 0.3 N m
    and 9.8e-9 at 1.2 N m for one step, 1.4e-6 over five steps, at eps 1e-4 (measured
    2026-09-07). The finite-difference step matters: at eps 1e-6 the quotients themselves
    are only self-consistent to 3e-5 at 1.2 N m, because the forward Newton solve (1e-12
    tolerance) stops after 76 to 85 iterations across the stencil and the loss jumps by the
    size of the last iterate; eps 1e-4 averages that out while the truncation error stays
    below 1e-8 (the loss is smooth in the torque). These tests assert exactness at that
    resolution, with a margin, and that the torque gradients are not trivially zero.
    """

    FD_EPS = 1e-4

    def _torque_adjoint_and_fd(self, scene_fn, loss_cls, amplitude, num_steps=1, newton_outer=False):
        diffsim_torch = _make_bridge_module()
        scene, cube = scene_fn()
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        if newton_outer:
            params = diffsim.get_back_propagation_solver_params(scene)
            params.use_newton_outer_solver = True
            diffsim.set_back_propagation_solver_params(scene, params)
        bridge = diffsim_torch.TorchRollout(
            scene,
            dt=DT,
            num_steps=num_steps,
            force_actors=[cube],
            terminal_losses=[loss_cls(cube)],
        )
        self.addCleanup(bridge.close)
        base = np.zeros((num_steps, 6))
        base[:, 3:] = amplitude * np.cos(np.arange(3 * num_steps)).reshape(num_steps, 3)
        f0 = torch.tensor(base, dtype=torch.float64, requires_grad=True)
        bridge(forces=f0).backward()
        adjoint = f0.grad.numpy()[:, 3:].copy()
        fd = np.zeros_like(adjoint)
        for k in range(num_steps):
            for i, dof in enumerate(range(3, 6)):
                values = []
                for sign in (+1.0, -1.0):
                    perturbed = base.copy()
                    perturbed[k, dof] += sign * self.FD_EPS
                    values.append(
                        bridge(
                            forces=torch.tensor(perturbed, dtype=torch.float64)
                        ).item()
                    )
                fd[k, i] = (values[0] - values[1]) / (2 * self.FD_EPS)
        self.assertGreater(float(np.abs(fd).max()), 1e-6, "vacuous: the torque does not move the loss")
        return adjoint, fd

    @staticmethod
    def _relative(adjoint, fd):
        # Relative to the block's largest entry: components the scene's symmetry leaves near
        # zero would otherwise compare rounding with rounding.
        return np.abs(adjoint - fd) / max(float(np.abs(fd).max()), 1e-14)

    def test_free_cube_moderate_torque_is_exact(self) -> None:
        adjoint, fd = self._torque_adjoint_and_fd(
            scenes.rigid_free, QuaternionErrorLoss, amplitude=0.3
        )
        self.assertLess(float(self._relative(adjoint, fd).max()), 1e-8, (adjoint, fd))  # 6.5e-10

    def test_free_cube_large_torque_is_exact(self) -> None:
        # 1.2 N m rotates this cube by 2.3e-3 rad per step; measured 9.8e-9 for one step and
        # 1.4e-6 for five (8.6e-4 / 2.0e-3 without the chart term).
        adjoint, fd = self._torque_adjoint_and_fd(
            scenes.rigid_free, QuaternionErrorLoss, amplitude=1.2
        )
        self.assertLess(float(self._relative(adjoint, fd).max()), 1e-7, (adjoint, fd))
        adjoint, fd = self._torque_adjoint_and_fd(
            scenes.rigid_free, QuaternionErrorLoss, amplitude=1.2, num_steps=5
        )
        self.assertLess(float(self._relative(adjoint, fd).max()), 1e-5, (adjoint, fd))

    def test_newton_outer_solver_carries_the_term_too(self) -> None:
        adjoint, fd = self._torque_adjoint_and_fd(
            scenes.rigid_free, QuaternionErrorLoss, amplitude=1.2, newton_outer=True
        )
        self.assertLess(float(self._relative(adjoint, fd).max()), 1e-7, (adjoint, fd))  # as Krylov

    def test_contact_scene_torque_gradients(self) -> None:
        """A cube sliding on the ground with a torque: the torque gradients of a translation
        loss are small (1e-8 to 1e-7, the torque barely moves the cube) and the adjoint
        reproduces them: 1.5e-11 absolute for one step and 2.2e-10 over five, 5e-5 relative
        to the block's largest entry (the finite differences themselves agree between eps
        1e-4 and 1e-5 to 5e-6 and 7e-6)."""
        adjoint, fd = self._torque_adjoint_and_fd(
            lambda: scenes.rigid_on_plane("rich"), TranslationErrorLoss, amplitude=0.3
        )
        self.assertLess(float(np.abs(adjoint - fd).max()), 2e-10)
        self.assertLess(float(self._relative(adjoint, fd).max()), 5e-4, (adjoint, fd))
        adjoint, fd = self._torque_adjoint_and_fd(
            lambda: scenes.rigid_on_plane("rich"), TranslationErrorLoss, amplitude=0.3, num_steps=5
        )
        self.assertLess(float(np.abs(adjoint - fd).max()), 2e-9)
        self.assertLess(float(self._relative(adjoint, fd).max()), 5e-4, (adjoint, fd))


class GradcheckSoftTest(unittest.TestCase):
    """Soft cube on a static plane: initial nodal state + gravity + contact."""

    def test_gradcheck_initial_state_gravity_contact(self) -> None:
        diffsim_torch = _make_bridge_module()
        num_steps = 3
        scene, cube = scenes.soft_cube_on_plane("rich")
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        bridge = diffsim_torch.TorchRollout(
            scene,
            dt=DT,
            num_steps=num_steps,
            contact_actors=[cube],
            initial_state_actors=[cube],
            differentiate_gravity=True,
            terminal_losses=[DisplacementErrorLoss(cube)],
        )
        self.addCleanup(bridge.close)
        num_dofs = cube.get_num_dofs()
        self.assertEqual(bridge.initial_state_size, 2 * num_dofs)

        contact_base = torch.tensor([[1e8, 0.4, 0.1, 5.0]], dtype=torch.float64)

        def f(initial_states, contact_scale, gravity):
            return bridge(
                initial_states=initial_states,
                contact_params=contact_base * (1.0 + 0.05 * contact_scale),
                gravity=gravity,
            )

        u0 = np.zeros(num_dofs)
        v0 = np.tile(np.array([0.3, 0.0, 0.0]), num_dofs // 3)
        initial_states = torch.tensor(
            np.concatenate([u0, v0]), dtype=torch.float64, requires_grad=True
        )
        contact_scale = torch.zeros((1, 4), dtype=torch.float64, requires_grad=True)
        gravity = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64, requires_grad=True)
        self.assertTrue(
            torch.autograd.gradcheck(
                f,
                (initial_states, contact_scale, gravity),
                eps=1e-6,
                atol=1e-8,
                rtol=1e-4,
            )
        )
        self.assertTrue(bridge.last_result.fd_valid)
        # Vacuousness: the initial-state gradient must be non-trivial.
        loss = f(initial_states, contact_scale, gravity)
        loss.backward()
        self.assertGreater(float(initial_states.grad.abs().max()), 0.0)

    def test_gradcheck_soft_materials(self) -> None:
        diffsim_torch = _make_bridge_module()
        num_steps = 3
        scene, cube = scenes.soft_cube_on_plane("rich", mass_damping=1.0)
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        bridge = diffsim_torch.TorchRollout(
            scene,
            dt=DT,
            num_steps=num_steps,
            soft_material_actors=[cube],
            terminal_losses=[DisplacementErrorLoss(cube)],
        )
        self.addCleanup(bridge.close)
        params = cube.get_soft_material_params()
        base = torch.tensor(
            [[
                params.neo_hookean.youngs_modulus,
                params.neo_hookean.poisson_ratio,
                params.density,
                params.mass_damping_coefficient,
            ]],
            dtype=torch.float64,
        )
        self.assertTrue(bool((base > 0).all()), "every field must be strictly positive")

        # Reparameterize so gradcheck's uniform 1e-6 step is well scaled for
        # every field (Young's modulus is 1e5 Pa, Poisson's ratio 0.45).
        def f(scale):
            return bridge(soft_materials=base * (1.0 + 0.05 * scale))

        scale = torch.zeros((1, 4), dtype=torch.float64, requires_grad=True)
        self.assertTrue(
            torch.autograd.gradcheck(f, (scale,), eps=1e-6, atol=1e-8, rtol=1e-4)
        )
        self.assertTrue(bridge.last_result.fd_valid)
        loss = f(scale)
        loss.backward()
        self.assertTrue(bool((scale.grad.abs() > 0).all()), "test is vacuous")

    def test_soft_actor_group_contracts(self) -> None:
        diffsim_torch = _make_bridge_module()
        scene, cube = scenes.soft_cube_on_plane("rich")
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        losses = [DisplacementErrorLoss(cube)]
        with self.assertRaisesRegex(ValueError, "no external forces"):
            diffsim_torch.TorchRollout(
                scene, dt=DT, num_steps=2, force_actors=[cube], terminal_losses=losses
            )
        with self.assertRaisesRegex(ValueError, "density"):
            diffsim_torch.TorchRollout(
                scene, dt=DT, num_steps=2, density_actors=[cube], terminal_losses=losses
            )
        rigid_scene, rigid = scenes.rigid_on_plane("rich")
        self.addCleanup(physics.destroy_scene, rigid_scene)
        configure_for_differentiability(rigid_scene)
        with self.assertRaisesRegex(ValueError, "not a soft actor"):
            diffsim_torch.TorchRollout(
                rigid_scene,
                dt=DT,
                num_steps=2,
                soft_material_actors=[rigid],
                terminal_losses=[TranslationErrorLoss(rigid)],
            )

    def test_rigid_initial_state_group_not_implemented(self) -> None:
        diffsim_torch = _make_bridge_module()
        scene, cube = scenes.rigid_on_plane("rich")
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        with self.assertRaisesRegex(NotImplementedError, "soft actors only"):
            diffsim_torch.TorchRollout(
                scene,
                dt=DT,
                num_steps=2,
                initial_state_actors=[cube],
                terminal_losses=[TranslationErrorLoss(cube)],
            )


class ContractTest(unittest.TestCase):
    def _bridge(self):
        diffsim_torch = _make_bridge_module()
        scene, cube = scenes.rigid_on_plane("rich")
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        bridge = diffsim_torch.TorchRollout(
            scene,
            dt=DT,
            num_steps=2,
            density_actors=[cube],
            terminal_losses=[TranslationErrorLoss(cube)],
        )
        self.addCleanup(bridge.close)
        return bridge

    def test_missing_declared_group(self) -> None:
        bridge = self._bridge()
        with self.assertRaisesRegex(ValueError, "densities is required"):
            bridge()

    def test_undeclared_group_rejected(self) -> None:
        bridge = self._bridge()
        with self.assertRaisesRegex(ValueError, "not declared"):
            bridge(
                densities=torch.tensor([1000.0], dtype=torch.float64),
                gravity=torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64),
            )

    def test_wrong_dtype_rejected(self) -> None:
        bridge = self._bridge()
        with self.assertRaisesRegex(TypeError, "float64"):
            bridge(densities=torch.tensor([1000.0], dtype=torch.float32))

    def test_wrong_shape_rejected(self) -> None:
        bridge = self._bridge()
        with self.assertRaisesRegex(ValueError, "shape"):
            bridge(densities=torch.tensor([[1000.0]], dtype=torch.float64))

    def test_static_contact_actor_rejected(self) -> None:
        diffsim_torch = _make_bridge_module()
        scene, cube = scenes.rigid_on_plane("rich")
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        ground = None

        def visit(actor):
            nonlocal ground
            if actor.is_static():
                ground = actor

        scene.for_each_actor(visit)
        self.assertIsNotNone(ground)
        with self.assertRaisesRegex(ValueError, "static"):
            diffsim_torch.TorchRollout(
                scene,
                dt=DT,
                num_steps=2,
                contact_actors=[ground],
                terminal_losses=[TranslationErrorLoss(cube)],
            )

    def test_closed_bridge_rejects_calls(self) -> None:
        bridge = self._bridge()
        bridge.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            bridge(densities=torch.tensor([1000.0], dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()


class PolicyRolloutTest(unittest.TestCase):
    """Closed-loop policy gradients: a linear policy on the pendulum maps the observed
    joint pose (and, with history 2, the previous one) to the controller targets; the
    gradients with respect to the policy weights and biases must match central finite
    differences of the closed-loop rollout, which include the feedback path.
    """

    FD_EPS = 1e-6
    TOL = 1e-5

    def _closed_loop(self, history: int, running: bool, time_feature: bool = False):
        diffsim_torch = _make_bridge_module()
        scene, chain = scenes.pendulum(with_controller=True)
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        num_dofs = chain.get_num_dofs()
        num_steps = 6
        pose0 = np.zeros(num_dofs)
        chain.get_articulated_pose(pose0)
        terminal = ArticulatedPoseErrorLoss(chain, ref=pose0 + 0.1)
        running_loss = ArticulatedPoseErrorLoss(chain, ref=pose0 - 0.05)
        input_size = history * num_dofs + (1 if time_feature else 0)
        policy = torch.nn.Linear(input_size, num_dofs).double()
        with torch.no_grad():
            weight = 0.1 * np.cos(np.arange(input_size * num_dofs, dtype=np.float64)).reshape(
                num_dofs, input_size
            )
            weight[:, :num_dofs] += 0.9 * np.eye(num_dofs)  # mostly "hold the current pose"
            policy.weight.copy_(torch.tensor(weight))
            policy.bias.copy_(torch.tensor(0.02 * np.arange(1, num_dofs + 1, dtype=np.float64)))
        rollout = diffsim_torch.PolicyRollout(
            scene,
            dt=DT,
            num_steps=num_steps,
            policy=policy,
            observations=[diffsim_torch.ArticulatedPoseObservation(chain)],
            control_actors=[chain],
            history=history,
            time_feature=time_feature,
            terminal_losses=[terminal],
            step_losses=(lambda step: [running_loss]) if running else None,
        )
        self.addCleanup(rollout.close)
        return scene, chain, policy, rollout, terminal, running_loss, num_steps, history, num_dofs

    def _fd_check(self, history: int, running: bool, time_feature: bool = False) -> None:
        scene, chain, policy, rollout, terminal, running_loss, num_steps, history, n = self._closed_loop(
            history, running, time_feature
        )
        loss = rollout()
        loss.backward()
        analytic = {name: p.grad.detach().numpy().copy() for name, p in policy.named_parameters()}
        state_init = scene.capture_state()

        def objective(weight: np.ndarray, bias: np.ndarray) -> float:
            scene.restore_state(state_init, False)
            obs = np.zeros(n)
            chain.get_articulated_pose(obs)
            hist = [obs.copy()] * history
            total = 0.0
            for step in range(num_steps):
                features = np.concatenate(hist[:history])
                if time_feature:
                    features = np.append(features, step / num_steps)
                u = weight @ features + bias
                chain.set_articulated_target_pose(np.ascontiguousarray(u))
                scene.step(DT)
                chain.get_articulated_pose(obs)
                hist.insert(0, obs.copy())
                if running:
                    total += running_loss.value()
            return total + terminal.value()

        weight = policy.weight.detach().numpy().copy()
        bias = policy.bias.detach().numpy().copy()
        loss_value = float(loss.detach())
        self.assertAlmostEqual(objective(weight, bias), loss_value, delta=1e-12 * max(1.0, abs(loss_value)))
        for name, array in (("weight", weight), ("bias", bias)):
            fd = np.zeros_like(array)
            for index in np.ndindex(array.shape):
                values = []
                for sign in (+1.0, -1.0):
                    saved = array[index]
                    array[index] = saved + sign * self.FD_EPS
                    values.append(objective(weight, bias))
                    array[index] = saved
                fd[index] = (values[0] - values[1]) / (2.0 * self.FD_EPS)
            rel = np.linalg.norm(analytic[name] - fd) / np.linalg.norm(fd)
            self.assertLessEqual(rel, self.TOL, f"{name} gradient mismatch:\n{analytic[name]}\n{fd}")
        scene.release_all_states()
        result = rollout.last_result
        self.assertTrue(result.fd_valid)
        self.assertEqual(result.steps_swept, num_steps)

    def test_linear_policy_history_1_vs_fd(self) -> None:
        self._fd_check(history=1, running=False)

    def test_linear_policy_history_2_running_loss_vs_fd(self) -> None:
        self._fd_check(history=2, running=True)

    def test_linear_policy_time_feature_vs_fd(self) -> None:
        self._fd_check(history=1, running=True, time_feature=True)

    def test_torque_policy_vs_fd(self) -> None:
        """A torque policy on a pendulum without a controller (joint torques from the joint
        pose), plus a mixed targets-and-torques policy on the controller pendulum."""
        diffsim_torch = _make_bridge_module()
        for with_controller in (False, True):
            scene, chain = scenes.pendulum(with_controller=with_controller)
            self.addCleanup(physics.destroy_scene, scene)
            configure_for_differentiability(scene)
            n = chain.get_num_dofs()
            pose0 = np.zeros(n)
            chain.get_articulated_pose(pose0)
            terminal = ArticulatedPoseErrorLoss(chain, ref=pose0 + 0.1)
            out_size = 2 * n if with_controller else n
            policy = torch.nn.Linear(n, out_size).double()
            with torch.no_grad():
                policy.weight.copy_(torch.tensor(0.3 * np.cos(np.arange(out_size * n, dtype=np.float64)).reshape(out_size, n)))
                policy.bias.copy_(torch.tensor(0.05 * np.arange(1, out_size + 1, dtype=np.float64)))
                if with_controller:
                    policy.weight[:n, :] += torch.eye(n, dtype=torch.float64)
            force_dofs = np.arange(n, dtype=np.int32)
            rollout = diffsim_torch.PolicyRollout(
                scene, dt=DT, num_steps=5, policy=policy,
                observations=[diffsim_torch.ArticulatedPoseObservation(chain)],
                control_actors=[chain] if with_controller else (), force_actors=[chain],
                terminal_losses=[terminal],
            )
            self.addCleanup(rollout.close)
            loss = rollout()
            loss.backward()
            analytic = {name: p.grad.detach().numpy().copy() for name, p in policy.named_parameters()}
            state_init = scene.capture_state()
            weight = policy.weight.detach().numpy().copy()
            bias = policy.bias.detach().numpy().copy()

            def objective() -> float:
                scene.restore_state(state_init, False)
                obs = np.zeros(n)
                for _ in range(5):
                    chain.get_articulated_pose(obs)
                    u = weight @ obs + bias
                    if with_controller:
                        chain.set_articulated_target_pose(np.ascontiguousarray(u[:n]))
                    chain.set_external_forces_on_dofs(force_dofs, np.ascontiguousarray(u[-n:]))
                    scene.step(DT)
                return terminal.value()

            self.assertAlmostEqual(objective(), float(loss.detach()), delta=1e-12)
            for name, array in (("weight", weight), ("bias", bias)):
                fd = np.zeros_like(array)
                for index in np.ndindex(array.shape):
                    values = []
                    for sign in (+1.0, -1.0):
                        saved = array[index]
                        array[index] = saved + sign * self.FD_EPS
                        values.append(objective())
                        array[index] = saved
                    fd[index] = (values[0] - values[1]) / (2.0 * self.FD_EPS)
                rel = np.linalg.norm(analytic[name] - fd) / np.linalg.norm(fd)
                self.assertLessEqual(rel, self.TOL, f"controller={with_controller} {name}: rel {rel}")
            scene.release_all_states()

    def test_contact_force_observation_policy_vs_fd(self) -> None:
        """A policy fed the pushed cube's total contact force (a tactile signal), the chain
        pose and the time drives the chain pushing the cube; the loss is the cube's final
        position. The initial observation is the probe step's force (a constant); after that
        the force of step k is a function of the controls up to k, and the loss gradient
        reaches the policy through it (the engine's contact-force adjoint, with the chain a
        moving collider). The force feedback gain is small enough for a smooth push - every
        forward step must converge - and the force columns of the weight carry gradient.
        Directional FD along the gradient at eps 1e-6 / 1e-7 (the force feedback makes the
        piecewise-smooth contact loss rough at 1e-5), plus the largest weight entries where FD
        is smooth; the replay reproduces the probe."""
        diffsim_torch = _make_bridge_module()
        scene, chain, cube = scenes.chain_pushing_cube()
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        n = chain.get_num_dofs()
        observations = [
            diffsim_torch.ContactForceObservation(cube),
            diffsim_torch.ArticulatedPoseObservation(chain),
        ]
        obs_size = sum(o.size for o in observations)
        num_steps = 30
        start = np.asarray(cube.get_center_of_mass_transform().translation, dtype=np.float64)
        terminal = TranslationErrorLoss(cube, ref=start + np.array([0.08, 0.0, 0.0]))
        policy = torch.nn.Linear(obs_size + 1, n).double()
        with torch.no_grad():
            policy.weight.copy_(
                torch.tensor(0.3 * np.cos(np.arange(n * (obs_size + 1), dtype=np.float64)).reshape(n, obs_size + 1))
            )
            policy.weight[:, :3] *= 1e-3  # the force is in newtons (tens), the targets in radians
            policy.weight[0, -1] = -1.2  # the time feature ramps joint 0 into the cube
            policy.weight[1, -1] = 0.0
            policy.bias.zero_()
        rollout = diffsim_torch.PolicyRollout(
            scene, dt=DT, num_steps=num_steps, policy=policy, observations=observations,
            control_actors=[chain], time_feature=True, terminal_losses=[terminal],
        )
        self.addCleanup(rollout.close)
        self.assertTrue(rollout.needs_probe_step)
        loss = rollout()
        loss.backward()
        self.assertTrue(rollout.last_result.fd_valid)
        params = list(policy.parameters())
        analytic = [p.grad.detach().numpy().copy() for p in params]
        values = [p.detach().numpy().copy() for p in params]
        state_init = rollout.initial_state

        def objective() -> float:
            # The policy is evaluated by torch exactly as the rollout does it: a numpy
            # `W @ obs + b` differs from torch's kernel by 1e-15 in the targets, which the
            # frictional contact amplifies to 1e-8 in this loss (a Newton solution branch).
            with torch.no_grad():
                for p, v in zip(params, values):
                    p.copy_(torch.tensor(v, dtype=torch.float64))
            scene.restore_state(state_init, False)
            rollout.probe_initial_observations()
            for step in range(num_steps):
                obs = np.concatenate([o.value() for o in observations] + [[step / num_steps]])
                with torch.no_grad():
                    targets = policy(torch.tensor(obs, dtype=torch.float64)).numpy()
                chain.set_articulated_target_pose(np.ascontiguousarray(targets))
                scene.step(DT)
                _assert_step_converged(self, scene, step)
            return terminal.value()

        self.assertAlmostEqual(objective(), float(loss.detach()), delta=1e-12)
        self.assertGreater(float(np.linalg.norm(observations[0].value())), 1.0, "the cube must be in contact")
        self.assertGreater(
            float(np.linalg.norm(np.asarray(cube.get_center_of_mass_transform().translation) - start)),
            0.01,
            "the chain must have pushed the cube",
        )
        grad_norm = float(np.sqrt(sum(float((g * g).sum()) for g in analytic)))
        direction = [g / grad_norm for g in analytic]

        def directional_fd(eps: float) -> float:
            pair = []
            for sign in (+1.0, -1.0):
                for v, d in zip(values, direction):
                    v += sign * eps * d
                pair.append(objective())
                for v, d in zip(values, direction):
                    v -= sign * eps * d
            return (pair[0] - pair[1]) / (2.0 * eps)

        fds = [directional_fd(eps) for eps in (1e-6, 1e-7)]
        self.assertLessEqual(abs(fds[0] - fds[1]) / abs(fds[0]), 1e-4, f"rough directional FD {fds}")
        self.assertLessEqual(abs(grad_norm - fds[0]) / abs(fds[0]), 1e-4, (grad_norm, fds))
        # The force columns of the weight carry gradient: the loss depends on the policy
        # through the observed contact force.
        force_columns = analytic[0][:, :3]
        self.assertGreater(float(np.abs(force_columns).max()), 0.0)
        rel_errors, skipped = {}, {}
        weight = values[0]
        for flat in np.argsort(-np.abs(analytic[0]).ravel())[:4]:
            index = np.unravel_index(int(flat), weight.shape)
            entry_fds = []
            for eps in (1e-6, 1e-7):
                pair = []
                for sign in (+1.0, -1.0):
                    saved = weight[index]
                    weight[index] = saved + sign * eps
                    pair.append(objective())
                    weight[index] = saved
                entry_fds.append((pair[0] - pair[1]) / (2.0 * eps))
            denom = max(abs(entry_fds[0]), 1e-30)
            fd_self = abs(entry_fds[0] - entry_fds[1]) / denom
            if fd_self > 1e-3:
                skipped[index] = fd_self
                continue
            rel_errors[index] = abs(analytic[0][index] - entry_fds[0]) / denom
        self.assertGreaterEqual(len(rel_errors), 2, f"too few smooth entries: {skipped}")
        self.assertLessEqual(max(rel_errors.values()), 1e-4, (rel_errors, skipped))

    def test_probe_step_leaves_the_state_and_reads_the_resting_force(self) -> None:
        """The probe fills the contact-force query without moving the scene: a cube resting on
        the ground reads its weight, and the state after the probe is the initial state."""
        diffsim_torch = _make_bridge_module()
        scene, cube = scenes.rigid_on_plane("coulomb", initial_velocity=(0.0, 0.0, 0.0))
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        for _ in range(50):  # settle
            scene.step(DT)
        observation = diffsim_torch.ContactForceObservation(cube)
        rollout = diffsim_torch.PolicyRollout(
            scene, dt=DT, num_steps=2, policy=torch.nn.Linear(3, 6).double(), observations=[observation],
            force_actors=[cube], terminal_losses=[TranslationErrorLoss(cube)],
        )
        self.addCleanup(rollout.close)
        before = np.asarray(cube.get_center_of_mass_transform().translation, dtype=np.float64).copy()
        rollout.probe_initial_observations()
        after = np.asarray(cube.get_center_of_mass_transform().translation, dtype=np.float64)
        np.testing.assert_allclose(after, before, atol=0.0)
        force = observation.value()
        weight = cube.get_mass() * 9.81
        self.assertAlmostEqual(force[2] / weight, 1.0, delta=1e-3, msg=str(force))

    def test_contact_force_observation_on_a_force_actor_vs_fd(self) -> None:
        """A cube sliding on the ground is both observed (its contact force) and driven (the
        policy's output is the external force on it, which shapes the normal force it
        observes). No torques. Directional FD along the gradient at eps 1e-6 / 1e-7, plus the
        largest weight entries where FD is smooth."""
        diffsim_torch = _make_bridge_module()
        scene, cube = scenes.rigid_on_plane("coulomb", initial_velocity=(0.5, 0.0, 0.0))
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        observations = [diffsim_torch.ContactForceObservation(cube)]
        num_steps = 25
        start = np.asarray(cube.get_center_of_mass_transform().translation, dtype=np.float64)
        terminal = TranslationErrorLoss(cube, ref=start + np.array([0.06, 0.0, 0.0]))
        policy = torch.nn.Linear(3 + 1, 6).double()
        with torch.no_grad():
            policy.weight.copy_(torch.tensor(0.1 * np.cos(np.arange(6 * 4, dtype=np.float64)).reshape(6, 4)))
            policy.weight[3:, :] = 0.0  # forces only
            policy.weight[0, -1] = 10.0  # a time-ramped push [N]
            policy.bias.zero_()
        rollout = diffsim_torch.PolicyRollout(
            scene, dt=DT, num_steps=num_steps, policy=policy, observations=observations,
            force_actors=[cube], time_feature=True, terminal_losses=[terminal],
        )
        self.addCleanup(rollout.close)
        loss = rollout()
        loss.backward()
        self.assertTrue(rollout.last_result.fd_valid)
        params = list(policy.parameters())
        analytic = [p.grad.detach().numpy().copy() for p in params]
        values = [p.detach().numpy().copy() for p in params]
        state_init = rollout.initial_state
        dofs = np.arange(6, dtype=np.int32)

        def objective() -> float:
            with torch.no_grad():
                for p, v in zip(params, values):
                    p.copy_(torch.tensor(v, dtype=torch.float64))
            scene.restore_state(state_init, False)
            rollout.probe_initial_observations()
            for step in range(num_steps):
                obs = np.concatenate([observations[0].value(), [step / num_steps]])
                with torch.no_grad():
                    forces = policy(torch.tensor(obs, dtype=torch.float64)).numpy()
                cube.set_external_forces_on_dofs(dofs, np.ascontiguousarray(forces))
                scene.step(DT)
                _assert_step_converged(self, scene, step)
            return terminal.value()

        self.assertAlmostEqual(objective(), float(loss.detach()), delta=1e-12)
        self.assertGreater(float(np.linalg.norm(observations[0].value())), 1.0, "the cube must be in contact")
        force_columns = analytic[0][:3, :3]
        self.assertGreater(float(np.abs(force_columns).max()), 0.0)
        grad_norm = float(np.sqrt(sum(float((g * g).sum()) for g in analytic)))
        direction = [g / grad_norm for g in analytic]

        def directional_fd(eps: float) -> float:
            pair = []
            for sign in (+1.0, -1.0):
                for v, d in zip(values, direction):
                    v += sign * eps * d
                pair.append(objective())
                for v, d in zip(values, direction):
                    v -= sign * eps * d
            return (pair[0] - pair[1]) / (2.0 * eps)

        fds = [directional_fd(eps) for eps in (1e-6, 1e-7)]
        self.assertLessEqual(abs(fds[0] - fds[1]) / abs(fds[0]), 1e-4, f"rough directional FD {fds}")
        self.assertLessEqual(abs(grad_norm - fds[0]) / abs(fds[0]), 1e-4, (grad_norm, fds))
        rel_errors, skipped = {}, {}
        weight = values[0]
        for flat in np.argsort(-np.abs(analytic[0]).ravel())[:4]:
            index = np.unravel_index(int(flat), weight.shape)
            entry_fds = []
            for eps in (1e-6, 1e-7):
                pair = []
                for sign in (+1.0, -1.0):
                    saved = weight[index]
                    weight[index] = saved + sign * eps
                    pair.append(objective())
                    weight[index] = saved
                entry_fds.append((pair[0] - pair[1]) / (2.0 * eps))
            denom = max(abs(entry_fds[0]), 1e-30)
            fd_self = abs(entry_fds[0] - entry_fds[1]) / denom
            if fd_self > 1e-3:
                skipped[index] = fd_self
                continue
            rel_errors[index] = abs(analytic[0][index] - entry_fds[0]) / denom
        self.assertGreaterEqual(len(rel_errors), 2, f"too few smooth entries: {skipped}")
        self.assertLessEqual(max(rel_errors.values()), 1e-4, (rel_errors, skipped))

    def test_contact_force_observation_on_a_soft_collider_vs_fd(self) -> None:
        """A rigid box resting on a soft cube that is an SDF collider, observed (its contact
        force, from both directions of the contact) and driven (the policy's output is the
        external force on it): directional FD along the gradient at eps 1e-6 / 1e-7."""
        diffsim_torch = _make_bridge_module()
        scene, soft, rigid = scenes.rigid_on_soft_collider("coulomb")
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        observations = [diffsim_torch.ContactForceObservation(rigid)]
        num_steps = 12
        start = np.asarray(rigid.get_center_of_mass_transform().translation, dtype=np.float64)
        terminal = TranslationErrorLoss(rigid, ref=start + np.array([0.04, 0.0, -0.01]))
        policy = torch.nn.Linear(3 + 1, 6).double()
        with torch.no_grad():
            policy.weight.copy_(torch.tensor(0.05 * np.cos(np.arange(6 * 4, dtype=np.float64)).reshape(6, 4)))
            policy.weight[3:, :] = 0.0
            policy.weight[0, -1] = 4.0
            policy.bias.zero_()
        rollout = diffsim_torch.PolicyRollout(
            scene, dt=DT, num_steps=num_steps, policy=policy, observations=observations,
            force_actors=[rigid], time_feature=True, terminal_losses=[terminal],
        )
        self.addCleanup(rollout.close)
        loss = rollout()
        loss.backward()
        self.assertTrue(rollout.last_result.fd_valid)
        params = list(policy.parameters())
        analytic = [p.grad.detach().numpy().copy() for p in params]
        values = [p.detach().numpy().copy() for p in params]
        state_init = rollout.initial_state
        dofs = np.arange(6, dtype=np.int32)
        self.assertGreater(float(np.abs(analytic[0][:3, :3]).max()), 0.0)

        def objective() -> float:
            with torch.no_grad():
                for p, v in zip(params, values):
                    p.copy_(torch.tensor(v, dtype=torch.float64))
            scene.restore_state(state_init, False)
            rollout.probe_initial_observations()
            for step in range(num_steps):
                obs = np.concatenate([observations[0].value(), [step / num_steps]])
                with torch.no_grad():
                    forces = policy(torch.tensor(obs, dtype=torch.float64)).numpy()
                rigid.set_external_forces_on_dofs(dofs, np.ascontiguousarray(forces))
                scene.step(DT)
                _assert_step_converged(self, scene, step)
            return terminal.value()

        self.assertAlmostEqual(objective(), float(loss.detach()), delta=1e-12)
        self.assertGreater(float(np.linalg.norm(observations[0].value())), 1.0, "the box must be in contact")
        grad_norm = float(np.sqrt(sum(float((g * g).sum()) for g in analytic)))
        direction = [g / grad_norm for g in analytic]

        def directional_fd(eps: float) -> float:
            pair = []
            for sign in (+1.0, -1.0):
                for v, d in zip(values, direction):
                    v += sign * eps * d
                pair.append(objective())
                for v, d in zip(values, direction):
                    v -= sign * eps * d
            return (pair[0] - pair[1]) / (2.0 * eps)

        fds = [directional_fd(eps) for eps in (1e-6, 1e-7)]
        self.assertLessEqual(abs(fds[0] - fds[1]) / abs(fds[0]), 1e-4, f"rough directional FD {fds}")
        self.assertLessEqual(abs(grad_norm - fds[0]) / abs(fds[0]), 1e-4, (grad_norm, fds))

    def test_contact_force_observation_with_substeps_vs_fd(self) -> None:
        """The tactile observation over split steps. The driver's failure-adaptive
        substepping is replaced by a deterministic split of every step into two half steps,
        so the records carry dt / 2 and the observation after a step is the force of its last
        half step; the gradient must match the directional FD (eps 1e-6 / 1e-7) of a replay
        taking the same half steps."""
        diffsim_torch = _make_bridge_module()
        scene, chain, cube = scenes.chain_pushing_cube()
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        n = chain.get_num_dofs()
        observations = [
            diffsim_torch.ContactForceObservation(cube),
            diffsim_torch.ArticulatedPoseObservation(chain),
        ]
        obs_size = sum(o.size for o in observations)
        num_steps = 15
        start = np.asarray(cube.get_center_of_mass_transform().translation, dtype=np.float64)
        terminal = TranslationErrorLoss(cube, ref=start + np.array([0.08, 0.0, 0.0]))
        policy = torch.nn.Linear(obs_size + 1, n).double()
        with torch.no_grad():
            policy.weight.copy_(
                torch.tensor(0.3 * np.cos(np.arange(n * (obs_size + 1), dtype=np.float64)).reshape(n, obs_size + 1))
            )
            policy.weight[:, :3] *= 1e-3
            policy.weight[0, -1] = -1.2
            policy.weight[1, -1] = 0.0
            policy.bias.zero_()

        def two_halves(scene_, dt, max_levels, residual_tolerance, step, on_substep):
            for _ in range(2):
                pre = scene_.capture_state()
                scene_.step(dt / 2.0)
                on_substep(pre, scene_.capture_state(), dt / 2.0)

        rollout = diffsim_torch.PolicyRollout(
            scene, dt=DT, num_steps=num_steps, policy=policy, observations=observations,
            control_actors=[chain], time_feature=True, terminal_losses=[terminal],
            max_substep_levels=1, substep_residual_tolerance=1e-6,
        )
        self.addCleanup(rollout.close)
        with mock.patch.object(diffsim_torch, "step_with_substeps", two_halves):
            loss = rollout()
            loss.backward()
        self.assertEqual(rollout.last_result.steps_swept, 2 * num_steps)
        self.assertEqual(len(rollout.last_result.split_steps), num_steps)
        self.assertTrue(rollout.last_result.fd_valid)
        params = list(policy.parameters())
        analytic = [p.grad.detach().numpy().copy() for p in params]
        values = [p.detach().numpy().copy() for p in params]
        state_init = rollout.initial_state
        self.assertGreater(float(np.abs(analytic[0][:, :3]).max()), 0.0)

        def objective() -> float:
            with torch.no_grad():
                for p, v in zip(params, values):
                    p.copy_(torch.tensor(v, dtype=torch.float64))
            scene.restore_state(state_init, False)
            rollout.probe_initial_observations()
            for step in range(num_steps):
                obs = np.concatenate([o.value() for o in observations] + [[step / num_steps]])
                with torch.no_grad():
                    targets = policy(torch.tensor(obs, dtype=torch.float64)).numpy()
                chain.set_articulated_target_pose(np.ascontiguousarray(targets))
                for _ in range(2):
                    scene.step(DT / 2.0)
                    _assert_step_converged(self, scene, step)
            return terminal.value()

        self.assertAlmostEqual(objective(), float(loss.detach()), delta=1e-12)
        grad_norm = float(np.sqrt(sum(float((g * g).sum()) for g in analytic)))
        direction = [g / grad_norm for g in analytic]

        def directional_fd(eps: float) -> float:
            pair = []
            for sign in (+1.0, -1.0):
                for v, d in zip(values, direction):
                    v += sign * eps * d
                pair.append(objective())
                for v, d in zip(values, direction):
                    v -= sign * eps * d
            return (pair[0] - pair[1]) / (2.0 * eps)

        fds = [directional_fd(eps) for eps in (1e-6, 1e-7)]
        self.assertLessEqual(abs(fds[0] - fds[1]) / abs(fds[0]), 1e-4, f"rough directional FD {fds}")
        self.assertLessEqual(abs(grad_norm - fds[0]) / abs(fds[0]), 1e-4, (grad_norm, fds))

    def test_contact_force_observation_on_a_link_vs_fd(self) -> None:
        """A static actor is refused at construction; an articulated link is observed like a
        standalone body. The policy is fed the contact force on the chain's lower link (the
        force the pushed cube exerts on the pusher, a tactile fingertip) and the chain pose;
        the loss is the cube's final position. Directional FD at eps 1e-7 / 1e-8: the
        difference quotient of this loss is converged from 1e-7 on (0.52758 at 1e-7 and 1e-8,
        0.52766 at 1e-6, 0.503 at 1e-5), the analytic gradient 0.5275772."""
        diffsim_torch = _make_bridge_module()
        scene, chain, cube = scenes.chain_pushing_cube()
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        actors = {}
        scene.for_each_actor(lambda actor: actors.__setitem__(actor.get_name(), actor))
        with self.assertRaisesRegex(ValueError, "static"):
            diffsim_torch.ContactForceObservation(actors["ground"])
        with self.assertRaisesRegex(ValueError, "not a rigid actor"):
            diffsim_torch.ContactForceObservation(chain)
        link = actors["chain/l1"]
        self.assertEqual(link.get_type(), physics.ActorType.RIGID)
        n = chain.get_num_dofs()
        observations = [
            diffsim_torch.ContactForceObservation(link),
            diffsim_torch.ArticulatedPoseObservation(chain),
        ]
        obs_size = sum(o.size for o in observations)
        num_steps = 30
        start = np.asarray(cube.get_center_of_mass_transform().translation, dtype=np.float64)
        terminal = TranslationErrorLoss(cube, ref=start + np.array([0.08, 0.0, 0.0]))
        policy = torch.nn.Linear(obs_size + 1, n).double()
        with torch.no_grad():
            policy.weight.copy_(
                torch.tensor(0.3 * np.cos(np.arange(n * (obs_size + 1), dtype=np.float64)).reshape(n, obs_size + 1))
            )
            policy.weight[:, :3] *= 1e-3
            policy.weight[0, -1] = -1.2
            policy.weight[1, -1] = 0.0
            policy.bias.zero_()
        rollout = diffsim_torch.PolicyRollout(
            scene, dt=DT, num_steps=num_steps, policy=policy, observations=observations,
            control_actors=[chain], time_feature=True, terminal_losses=[terminal],
        )
        self.addCleanup(rollout.close)
        loss = rollout()
        loss.backward()
        self.assertTrue(rollout.last_result.fd_valid)
        params = list(policy.parameters())
        analytic = [p.grad.detach().numpy().copy() for p in params]
        values = [p.detach().numpy().copy() for p in params]
        state_init = rollout.initial_state
        self.assertGreater(float(np.abs(analytic[0][:, :3]).max()), 0.0, "the link's force must carry gradient")

        def objective() -> float:
            with torch.no_grad():
                for p, v in zip(params, values):
                    p.copy_(torch.tensor(v, dtype=torch.float64))
            scene.restore_state(state_init, False)
            rollout.probe_initial_observations()
            for step in range(num_steps):
                obs = np.concatenate([o.value() for o in observations] + [[step / num_steps]])
                with torch.no_grad():
                    targets = policy(torch.tensor(obs, dtype=torch.float64)).numpy()
                chain.set_articulated_target_pose(np.ascontiguousarray(targets))
                scene.step(DT)
                _assert_step_converged(self, scene, step)
            return terminal.value()

        self.assertAlmostEqual(objective(), float(loss.detach()), delta=1e-12)
        self.assertGreater(float(np.linalg.norm(observations[0].value())), 1.0, "the link must be in contact")
        grad_norm = float(np.sqrt(sum(float((g * g).sum()) for g in analytic)))
        direction = [g / grad_norm for g in analytic]

        def directional_fd(eps: float) -> float:
            pair = []
            for sign in (+1.0, -1.0):
                for v, d in zip(values, direction):
                    v += sign * eps * d
                pair.append(objective())
                for v, d in zip(values, direction):
                    v -= sign * eps * d
            return (pair[0] - pair[1]) / (2.0 * eps)

        fds = [directional_fd(eps) for eps in (1e-7, 1e-8)]
        self.assertLessEqual(abs(fds[0] - fds[1]) / abs(fds[0]), 1e-4, f"rough directional FD {fds}")
        self.assertLessEqual(abs(grad_norm - fds[0]) / abs(fds[0]), 1e-4, (grad_norm, fds))

    def test_orientation_observation_policy_vs_fd(self) -> None:
        """A policy fed the pushed cube's orientation quaternion (and the chain pose and time)
        drives the controlled chain pushing a rigid cube: the off-center push spins the cube,
        so the observed orientation depends on the earlier controls and the loss (the cube's
        final position) sees the policy through it. No external torques. Directional FD along the gradient at eps 1e-5 / 1e-6,
        plus the largest weight entries where FD is smooth."""
        diffsim_torch = _make_bridge_module()
        scene, chain, cube = scenes.chain_pushing_cube()
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        n = chain.get_num_dofs()
        observations = [
            diffsim_torch.OrientationObservation(cube),
            diffsim_torch.ArticulatedPoseObservation(chain),
        ]
        obs_size = sum(o.size for o in observations)
        num_steps = 30
        start = np.asarray(cube.get_center_of_mass_transform().translation, dtype=np.float64)
        terminal = TranslationErrorLoss(cube, ref=start + np.array([0.08, 0.0, 0.0]))
        policy = torch.nn.Linear(obs_size + 1, n).double()
        with torch.no_grad():
            policy.weight.copy_(
                torch.tensor(0.3 * np.cos(np.arange(n * (obs_size + 1), dtype=np.float64)).reshape(n, obs_size + 1))
            )
            policy.weight[0, -1] = -1.2  # the time feature ramps joint 0 into the cube
            policy.weight[1, -1] = 0.0
            policy.bias.zero_()
        rollout = diffsim_torch.PolicyRollout(
            scene, dt=DT, num_steps=num_steps, policy=policy, observations=observations,
            control_actors=[chain], time_feature=True, terminal_losses=[terminal],
        )
        self.addCleanup(rollout.close)
        loss = rollout()
        loss.backward()
        self.assertTrue(rollout.last_result.fd_valid)
        params = list(policy.parameters())
        analytic = [p.grad.detach().numpy().copy() for p in params]
        state_init = scene.capture_state()
        values = [p.detach().numpy().copy() for p in params]

        def objective() -> float:
            scene.restore_state(state_init, False)
            for step in range(num_steps):
                obs = np.concatenate([o.value() for o in observations] + [[step / num_steps]])
                chain.set_articulated_target_pose(np.ascontiguousarray(values[0] @ obs + values[1]))
                scene.step(DT)
            return terminal.value()

        self.assertAlmostEqual(objective(), float(loss.detach()), delta=1e-12)
        quat = observations[0].value()
        self.assertGreater(float(np.abs(quat[:3]).max()), 1e-3, "the cube must have turned")
        grad_norm = float(np.sqrt(sum(float((g * g).sum()) for g in analytic)))
        direction = [g / grad_norm for g in analytic]

        def directional_fd(eps: float) -> float:
            pair = []
            for sign in (+1.0, -1.0):
                for v, d in zip(values, direction):
                    v += sign * eps * d
                pair.append(objective())
                for v, d in zip(values, direction):
                    v -= sign * eps * d
            return (pair[0] - pair[1]) / (2.0 * eps)

        fds = [directional_fd(eps) for eps in (1e-5, 1e-6)]
        self.assertLessEqual(abs(fds[0] - fds[1]) / abs(fds[0]), 1e-4, f"rough directional FD {fds}")
        self.assertLessEqual(abs(grad_norm - fds[0]) / abs(fds[0]), 1e-4, (grad_norm, fds))
        rel_errors, skipped = {}, {}
        weight = values[0]
        for flat in np.argsort(-np.abs(analytic[0]).ravel())[:4]:
            index = np.unravel_index(int(flat), weight.shape)
            entry_fds = []
            for eps in (1e-5, 1e-6):
                pair = []
                for sign in (+1.0, -1.0):
                    saved = weight[index]
                    weight[index] = saved + sign * eps
                    pair.append(objective())
                    weight[index] = saved
                entry_fds.append((pair[0] - pair[1]) / (2.0 * eps))
            denom = max(abs(entry_fds[0]), 1e-30)
            fd_self = abs(entry_fds[0] - entry_fds[1]) / denom
            if fd_self > 1e-3:
                skipped[index] = fd_self
                continue
            rel_errors[index] = abs(analytic[0][index] - entry_fds[0]) / denom
        self.assertGreaterEqual(len(rel_errors), 2, f"too few smooth entries: {skipped}")
        self.assertLessEqual(max(rel_errors.values()), 1e-4, (rel_errors, skipped))
        scene.release_all_states()

    def test_soft_observation_policy_vs_fd(self) -> None:
        """A policy fed the soft cube's centroid displacement, the chain pose and the time
        drives the chain pushing the cube (the time weight ramps joint 0 as the open-loop
        soft test does); the loss is the centroid's final displacement. Directional FD along
        the gradient at eps 1e-5 / 1e-6, plus the largest entries where FD is smooth."""
        diffsim_torch = _make_bridge_module()
        scene, chain, soft = scenes.chain_pushing_soft_cube()
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        n = chain.get_num_dofs()
        observations = [
            diffsim_torch.SoftCentroidObservation(soft),
            diffsim_torch.ArticulatedPoseObservation(chain),
        ]
        obs_size = sum(o.size for o in observations)
        num_nodes = soft.get_num_dofs() // 3
        goal = np.array([0.08, 0.0, -0.01])
        num_steps = 30

        def centroid() -> np.ndarray:
            return np.asarray(soft.get_displacements(), dtype=np.float64).reshape(num_nodes, 3).mean(axis=0)

        class CentroidLoss:
            def value(self) -> float:
                d = centroid() - goal
                return 0.5 * float(d @ d)

            def accumulate_output_grad(self) -> None:
                diffsim.get_displacements_backward(
                    soft, np.ascontiguousarray(np.tile((centroid() - goal) / num_nodes, num_nodes), dtype=real_dtype())
                )

        policy = torch.nn.Linear(obs_size + 1, n).double()
        with torch.no_grad():
            policy.weight.copy_(
                torch.tensor(0.3 * np.cos(np.arange(n * (obs_size + 1), dtype=np.float64)).reshape(n, obs_size + 1))
            )
            policy.weight[0, -1] = -1.2  # the time feature ramps joint 0 into the cube
            policy.weight[1, -1] = 0.0
            policy.bias.zero_()
        rollout = diffsim_torch.PolicyRollout(
            scene, dt=DT, num_steps=num_steps, policy=policy, observations=observations,
            control_actors=[chain], time_feature=True, terminal_losses=[CentroidLoss()],
        )
        self.addCleanup(rollout.close)
        loss = rollout()
        loss.backward()
        self.assertTrue(rollout.last_result.fd_valid)
        params = list(policy.parameters())
        analytic = [p.grad.detach().numpy().copy() for p in params]
        state_init = scene.capture_state()
        values = [p.detach().numpy().copy() for p in params]

        def objective() -> float:
            scene.restore_state(state_init, False)
            for step in range(num_steps):
                obs = np.concatenate([o.value() for o in observations] + [[step / num_steps]])
                chain.set_articulated_target_pose(np.ascontiguousarray(values[0] @ obs + values[1]))
                scene.step(DT)
            return CentroidLoss().value()

        self.assertAlmostEqual(objective(), float(loss.detach()), delta=1e-12)
        self.assertGreater(float(centroid()[0]), 0.01, "the cube must have been pushed")
        grad_norm = float(np.sqrt(sum(float((g * g).sum()) for g in analytic)))
        direction = [g / grad_norm for g in analytic]

        def directional_fd(eps: float) -> float:
            pair = []
            for sign in (+1.0, -1.0):
                for v, d in zip(values, direction):
                    v += sign * eps * d
                pair.append(objective())
                for v, d in zip(values, direction):
                    v -= sign * eps * d
            return (pair[0] - pair[1]) / (2.0 * eps)

        fds = [directional_fd(eps) for eps in (1e-5, 1e-6)]
        self.assertLessEqual(abs(fds[0] - fds[1]) / abs(fds[0]), 1e-4, f"rough directional FD {fds}")
        self.assertLessEqual(abs(grad_norm - fds[0]) / abs(fds[0]), 1e-4, (grad_norm, fds))
        # The largest weight entries, each at its own FD self-consistency level.
        rel_errors, skipped = {}, {}
        weight = values[0]
        for flat in np.argsort(-np.abs(analytic[0]).ravel())[:4]:
            index = np.unravel_index(int(flat), weight.shape)
            entry_fds = []
            for eps in (1e-5, 1e-6):
                pair = []
                for sign in (+1.0, -1.0):
                    saved = weight[index]
                    weight[index] = saved + sign * eps
                    pair.append(objective())
                    weight[index] = saved
                entry_fds.append((pair[0] - pair[1]) / (2.0 * eps))
            denom = max(abs(entry_fds[0]), 1e-30)
            fd_self = abs(entry_fds[0] - entry_fds[1]) / denom
            if fd_self > 1e-3:
                skipped[index] = fd_self
                continue
            rel_errors[index] = abs(analytic[0][index] - entry_fds[0]) / denom
        self.assertGreaterEqual(len(rel_errors), 2, f"too few smooth entries: {skipped}")
        self.assertLessEqual(max(rel_errors.values()), 1e-4, (rel_errors, skipped))
        scene.release_all_states()

    def test_nonlinear_policy_vs_fd(self) -> None:
        diffsim_torch = _make_bridge_module()
        scene, chain = scenes.pendulum(with_controller=True)
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        n = chain.get_num_dofs()
        pose0 = np.zeros(n)
        chain.get_articulated_pose(pose0)
        terminal = ArticulatedPoseErrorLoss(chain, ref=pose0 + 0.1)
        torch.manual_seed(0)
        policy = torch.nn.Sequential(torch.nn.Linear(n, 8), torch.nn.Tanh(), torch.nn.Linear(8, n)).double()
        with torch.no_grad():
            policy[2].weight.mul_(0.1)
            policy[2].bias.copy_(torch.tensor(pose0))
        rollout = diffsim_torch.PolicyRollout(
            scene, dt=DT, num_steps=5, policy=policy,
            observations=[diffsim_torch.ArticulatedPoseObservation(chain)],
            control_actors=[chain], terminal_losses=[terminal],
        )
        self.addCleanup(rollout.close)
        loss = rollout()
        loss.backward()
        params = [p for p in policy.parameters()]
        analytic = np.concatenate([p.grad.detach().numpy().ravel() for p in params])
        state_init = scene.capture_state()

        def objective() -> float:
            scene.restore_state(state_init, False)
            obs = np.zeros(n)
            for _ in range(5):
                chain.get_articulated_pose(obs)
                with torch.no_grad():
                    u = policy(torch.tensor(obs, dtype=torch.float64)).numpy()
                chain.set_articulated_target_pose(np.ascontiguousarray(u))
                scene.step(DT)
            return terminal.value()

        fd = []
        for p in params:
            flat = p.data.view(-1)
            for j in range(flat.numel()):
                saved = float(flat[j])
                values = []
                for sign in (+1.0, -1.0):
                    flat[j] = saved + sign * self.FD_EPS
                    values.append(objective())
                flat[j] = saved
                fd.append((values[0] - values[1]) / (2.0 * self.FD_EPS))
        fd = np.array(fd)
        scene.release_all_states()
        rel = np.linalg.norm(analytic - fd) / np.linalg.norm(fd)
        self.assertLessEqual(rel, self.TOL, f"policy gradient mismatch: rel {rel}")


class EngineContactForceAdjointTest(unittest.TestCase):
    """The engine's contact-force adjoint, ``get_contact_force_world_backward``, against the
    kinematic reference: for a free rigid body the contact force is ``m (v_k - v_{k-1}) / dt -
    m g - f_ext`` (an identity of the integrator), whose gradient uses only the exact
    center-of-mass position adjoint. Exact for contact with static colliders and moving ones
    alike since 2026-09-05, when two terms were added to the adjoint: the current-state SDF
    Hessian in the derivative of the penalty force (``d(N(d) g)/dp = N' g g^T + N H``; a grid
    SDF's interpolated gradient turns near edges, a plane's does not) and, for a dynamic
    collider, the rotation of the forces with the collider (``d(R_B f)/d delta = delta x R_B f``).
    Before: 3.7% against a static grid-SDF cube, 15% for a tilting rigid pusher and 26% for the
    controlled chain with Coulomb friction, all exact against a static plane."""

    NUM_STEPS = 20
    F_REF = np.array([5.0, 0.0, 30.0])

    def _engine_vs_kinematic(self, build, driven_name: str, apply_fn, inputs, driven_is_target: bool):
        """Relative difference of the gradient of ``0.5 |F_target - F_REF|^2`` (F the terminal
        total contact force on the target) with respect to the driven actor's per-step
        inputs: engine adjoint vs the kinematic reference."""
        from superdex.physics.diffsim_rollout import DifferentiableRollout

        n, dt = self.NUM_STEPS, DT
        f_ref = self.F_REF

        class QueryLoss:
            def __init__(self, target):
                self.target = target

            def value(self) -> float:
                d = np.asarray(self.target.get_contact_force_world()) - f_ref
                return 0.5 * float(d @ d)

            def accumulate_output_grad(self) -> None:
                d = np.asarray(self.target.get_contact_force_world(), dtype=np.float64) - f_ref
                diffsim.get_contact_force_world_backward(self.target, np.ascontiguousarray(d, dtype=real_dtype()))

        def block(grads):
            return (grads.control_targets if grads.control_targets is not None else grads.external_forces).copy()

        scene, driven, target = build()
        result = DifferentiableRollout(scene, dt=dt, num_steps=n).run(
            apply_inputs=lambda k: apply_fn(driven, inputs[k]), terminal_losses=[QueryLoss(target)]
        )
        g_engine = block(result.gradients[driven_name])
        scene.release_all_states()
        physics.destroy_scene(scene)

        scene, driven, target = build()
        m = target.get_mass()
        gravity = np.asarray(scene.get_gravity(), dtype=np.float64)
        positions = {}
        pos = lambda: np.asarray(target.get_center_of_mass_transform().translation, dtype=np.float64).copy()  # noqa: E731

        def apply(k):
            if k > 0:
                positions[k - 1] = pos()
            apply_fn(driven, inputs[k])

        def force():
            return m * (positions[n - 1] - 2 * positions[n - 2] + positions[n - 3]) / dt**2 - m * gravity - f_ext_last

        f_ext_last = inputs[n - 1][:3] if driven_is_target else np.zeros(3)

        class Recorder:
            def value(self) -> float:
                positions[n - 1] = pos()
                return 0.0

            def accumulate_output_grad(self) -> None:
                pass

        class KinematicLoss:
            def __init__(self, k, coefficient):
                self.k, self.coefficient = k, coefficient

            def value(self) -> float:
                d = force() - f_ref
                return 0.5 * float(d @ d) if self.k == n - 1 else 0.0

            def accumulate_output_grad(self) -> None:
                grad = np.zeros(7, dtype=real_dtype())
                grad[:3] = self.coefficient * m / dt**2 * (force() - f_ref)
                diffsim.get_center_of_mass_transform_backward(target, grad)

        coefficients = {n - 2: -2.0, n - 3: 1.0}
        result = DifferentiableRollout(scene, dt=dt, num_steps=n).run(
            apply_inputs=apply,
            step_losses=lambda k: [KinematicLoss(k, coefficients[k])] if k in coefficients else [],
            terminal_losses=[Recorder(), KinematicLoss(n - 1, 1.0)],
        )
        g_kinematic = block(result.gradients[driven_name])
        if driven_is_target:
            # The direct term of the force in the balance, -f_ext, at the terminal step.
            g_kinematic[:3, n - 1] -= force() - f_ref
        scene.release_all_states()
        physics.destroy_scene(scene)
        return float(np.linalg.norm(g_engine - g_kinematic) / np.linalg.norm(g_kinematic))

    def _pusher(self, friction: str):
        contact = scenes.contact_params(friction)
        scene = physics.create_scene("pusher")
        scene.set_gravity([0.0, 0.0, -9.81])
        scene.create_rigid_actor(
            name="ground", shape=physics.create_plane_shape(normal=[0, 0, 1], distance=0.0),
            is_static=True, contact=contact,
        )
        half = float(np.abs(scenes.CUBE_COORDS).max())
        pusher = scene.create_rigid_actor(
            name="pusher", shape=scenes.cube_shape(), density=1000.0, contact=contact,
            world_from_local=physics.TransformRT([0.0, 0.0, half + 0.03]),
        )
        target = scene.create_rigid_actor(
            name="target", shape=scenes.cube_shape(), density=1000.0, contact=contact,
            world_from_local=physics.TransformRT([2 * half + 0.002, 0.0, half]),
        )
        configure_for_differentiability(scene)
        target.register_query(physics.QueryType.TOTAL_CONTACT_FORCE)
        return scene, pusher, target

    def test_static_collider_is_exact(self) -> None:
        """A cube sliding on the static ground (a plane), driven by external forces on itself."""
        def build():
            scene, cube = scenes.rigid_on_plane("coulomb", initial_velocity=(0.5, 0.0, 0.0))
            configure_for_differentiability(scene)
            cube.register_query(physics.QueryType.TOTAL_CONTACT_FORCE)
            return scene, cube, cube

        dofs = np.arange(6, dtype=np.int32)
        forces = np.zeros((self.NUM_STEPS, 6))
        forces[:, 0] = np.linspace(0.0, 20.0, self.NUM_STEPS)
        rel = self._engine_vs_kinematic(
            build, "cube", lambda a, u: a.set_external_forces_on_dofs(dofs, np.ascontiguousarray(u)), forces, True
        )
        self.assertLessEqual(rel, 1e-6, rel)

    def test_static_grid_sdf_and_mesh_colliders_are_exact(self) -> None:
        """A flying cube (pushed and tilted by external forces) hitting a static cube: the
        contact meets the collider near its edges, where the gradient of the signed distance
        turns - the interpolated gradient of a grid SDF (measured 3.7% before the Hessian
        term, 3e-7 / 1e-8 after) and the closest-feature gradient of a triangle-mesh collider,
        whose Hessian is analytic by feature."""
        def build(friction, collider_type):
            contact = scenes.contact_params(friction)
            scene = physics.create_scene("wall")
            scene.set_gravity([0.0, 0.0, -9.81])
            half = float(np.abs(scenes.CUBE_COORDS).max())
            scene.create_rigid_actor(
                name="wall", shape=scenes.cube_shape(), is_static=True, contact=contact,
                collider_type=collider_type,
                world_from_local=physics.TransformRT([2 * half + 0.004, 0.0, 0.0]),
            )
            cube = scene.create_rigid_actor(name="cube", shape=scenes.cube_shape(), density=1000.0, contact=contact)
            configure_for_differentiability(scene)
            cube.register_query(physics.QueryType.TOTAL_CONTACT_FORCE)
            return scene, cube, cube

        dofs = np.arange(6, dtype=np.int32)
        for collider_type in (physics.ColliderType.SDF, physics.ColliderType.MESH):
            for friction in ("none", "coulomb"):
                forces = np.zeros((self.NUM_STEPS, 6))
                forces[:, 0] = 60.0
                forces[:, 4] = 3.0
                rel = self._engine_vs_kinematic(
                    lambda friction=friction, collider_type=collider_type: build(friction, collider_type),
                    "cube",
                    lambda a, u: a.set_external_forces_on_dofs(dofs, np.ascontiguousarray(u)), forces, True,
                )
                self.assertLessEqual(rel, 1e-6, (collider_type, friction, rel))

    def test_soft_body_colliders_are_exact(self) -> None:
        """The force on a rigid box resting on a soft cube, driven by external forces on the box.
        With the soft cube an SDF collider (scenes.rigid_on_soft_collider) the force has both
        directions of the contact: the box's samples against the cube's mapped SDF (the
        adjoint reaches the cube's nodes through the mapping and through the deformation
        gradient of its tetrahedra) and the cube's samples against the box's SDF (the
        samples' Jacobian w.r.t. the cube's nodes); with the cube a plain soft actor
        (scenes.soft_cube_under_rigid, its samples only) the second direction alone. Both
        against the kinematic identity of the box (implemented 2026-09-06; the query adjoint
        used to refuse the first and drop the second silently)."""
        dofs = np.arange(6, dtype=np.int32)
        for label, make in (
            ("soft SDF collider, both directions", lambda: scenes.rigid_on_soft_collider("coulomb")),
            ("soft samples only", lambda: scenes.soft_cube_under_rigid("coulomb", rigid_velocity=(0.0, 0.0, 0.0))),
        ):
            def build(make=make):
                scene, soft, rigid = make()
                configure_for_differentiability(scene)
                rigid.register_query(physics.QueryType.TOTAL_CONTACT_FORCE)
                return scene, rigid, rigid

            forces = np.zeros((self.NUM_STEPS, 6))
            forces[:, 0] = np.linspace(2.0, 8.0, self.NUM_STEPS)
            forces[:, 4] = 0.3
            rel = self._engine_vs_kinematic(
                build, "rigid", lambda a, u: a.set_external_forces_on_dofs(dofs, np.ascontiguousarray(u)), forces, True
            )
            self.assertLessEqual(rel, 1e-5, (label, rel))

    def test_rod_samples_are_exact(self) -> None:
        """The force on a rigid cube that a rod is dropped on (scenes.rod_onto_cube: the rod's
        centerline samples against the cube's SDF, the cube sliding on a frictionless ground),
        driven by external forces on the cube, against the cube's kinematic identity. The rod's
        samples reach its nodes (three displacements and a twist per node) through the segment
        interpolation, the same map as a soft body's samples (2026-09-06)."""
        def build():
            scene, rod, cube = scenes.rod_onto_cube(cube_velocity=(0.3, 0.0, 0.0))
            configure_for_differentiability(scene)
            cube.register_query(physics.QueryType.TOTAL_CONTACT_FORCE)
            return scene, cube, cube

        dofs = np.arange(6, dtype=np.int32)
        forces = np.zeros((self.NUM_STEPS, 6))
        forces[:, 0] = np.linspace(1.0, 4.0, self.NUM_STEPS)
        forces[:, 4] = 0.1
        rel = self._engine_vs_kinematic(
            build, "cube", lambda a, u: a.set_external_forces_on_dofs(dofs, np.ascontiguousarray(u)), forces, True
        )
        self.assertLessEqual(rel, 1e-5, rel)

    def test_moving_collider_is_exact(self) -> None:
        """A rigid pusher (pushed and tilted by external forces) against the target cube, with
        and without Coulomb friction (measured 12-15% before the rotation term, 5e-8 to 3e-7
        after); the controlled chain pushing the cube with Coulomb friction (26% before,
        2e-6 after)."""
        dofs = np.arange(6, dtype=np.int32)
        for friction in ("none", "coulomb"):
            forces = np.zeros((self.NUM_STEPS, 6))
            forces[:, 0] = 80.0
            forces[:, 4] = 15.0
            rel = self._engine_vs_kinematic(
                lambda friction=friction: self._pusher(friction), "pusher",
                lambda a, u: a.set_external_forces_on_dofs(dofs, np.ascontiguousarray(u)), forces, False,
            )
            self.assertLessEqual(rel, 1e-5, (friction, rel))

        def build_chain():
            scene, chain, cube = scenes.chain_pushing_cube("coulomb")
            configure_for_differentiability(scene)
            cube.register_query(physics.QueryType.TOTAL_CONTACT_FORCE)
            return scene, chain, cube

        targets = np.stack([np.linspace(0.0, -1.2, self.NUM_STEPS), np.zeros(self.NUM_STEPS)], axis=1)
        rel_chain = self._engine_vs_kinematic(
            build_chain, "chain", lambda a, u: a.set_articulated_target_pose(np.ascontiguousarray(u)), targets, False
        )
        self.assertLessEqual(rel_chain, 1e-5, rel_chain)
