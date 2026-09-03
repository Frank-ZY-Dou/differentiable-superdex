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
  is well-scaled (this also exercises chaining into an upstream torch graph).
  Torque gradients are excluded from gradcheck because of a known engine
  approximation, and that exclusion is licensed by two tests that PIN the
  approximation instead of hiding it (see
  :class:`TorqueGradientApproximationTest`);
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

import numpy as np
import superdex.physics as physics

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
)

_NUM_WORKER_THREADS = int(os.environ.get("SUPERDEX_DIFFSIM_TEST_THREADS", "0"))
DT = 0.01


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
        # Torque columns (DoFs 3-5) are held at a constant baseline instead
        # of being gradchecked: their gradients carry the engine's
        # O(per-step-rotation) merit-function approximation, which is
        # measured and pinned by TorqueGradientApproximationTest below.
        torque_baseline = torch.tensor(
            0.3 * np.cos(np.arange(3 * num_steps, dtype=np.float64)).reshape(
                num_steps, 3
            ),
            dtype=torch.float64,
        )

        def f(contact_scale, density_scale, linear_forces, gravity):
            return bridge(
                forces=torch.cat([linear_forces, torque_baseline], dim=1),
                gravity=gravity,
                contact_params=contact_base * (1.0 + 0.05 * contact_scale),
                densities=density_base * (1.0 + 0.05 * density_scale),
            )

        contact_scale = torch.zeros((1, 4), dtype=torch.float64, requires_grad=True)
        density_scale = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        linear_forces = torch.tensor(
            0.5
            * np.sin(np.arange(3 * num_steps, dtype=np.float64)).reshape(
                num_steps, 3
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
                (contact_scale, density_scale, linear_forces, gravity),
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


class TorqueGradientApproximationTest(unittest.TestCase):
    """Pins the engine's rotational-gradient approximation under external torques.

    With an external torque the rigid rotational gradients (dL/dtorque and, equally,
    dL/d(initial angular velocity)) carry a relative error proportional to the torque
    and independent of the angular velocity; without a torque they are exact to 1e-9
    at any per-step rotation. Half of it was the adjoint's Hessian-vector products
    transporting the constant torque term between rotation charts (removed on
    2026-09-03: external forces no longer enter those products); the remaining half
    is a symmetric term (the operator's asymmetry probe stays at 1e-9): measured
    2.2e-4 at 0.3 N*m and 8.6e-4 at 1.2 N*m on a free cube (before: 4.3e-4 and
    1.7e-3), and on the contact scene 6.5e-4 relative / 3.4e-10 absolute (before:
    2.4e-2 / 5.3e-10). These tests assert the deviation EXISTS (lower bound) and
    stays SMALL (upper bound): an engine fix that makes torque gradients exact flips
    the lower bound, prompting a documentation update and the reinstatement of
    torque columns into gradcheck.
    """

    def _torque_adjoint_and_fd(self, scene_fn, loss_cls, amplitude):
        diffsim_torch = _make_bridge_module()
        scene, cube = scene_fn()
        self.addCleanup(physics.destroy_scene, scene)
        configure_for_differentiability(scene)
        bridge = diffsim_torch.TorchRollout(
            scene,
            dt=DT,
            num_steps=1,
            force_actors=[cube],
            terminal_losses=[loss_cls(cube)],
        )
        self.addCleanup(bridge.close)
        base = np.zeros((1, 6))
        base[:, 3:] = amplitude * np.cos(np.arange(3)).reshape(1, 3)
        f0 = torch.tensor(base, dtype=torch.float64, requires_grad=True)
        bridge(forces=f0).backward()
        adjoint = f0.grad.numpy()[0, 3:].copy()
        fd = np.zeros(3)
        eps = 1e-6
        for i, dof in enumerate(range(3, 6)):
            values = []
            for sign in (+1.0, -1.0):
                perturbed = base.copy()
                perturbed[0, dof] += sign * eps
                values.append(
                    bridge(
                        forces=torch.tensor(perturbed, dtype=torch.float64)
                    ).item()
                )
            fd[i] = (values[0] - values[1]) / (2 * eps)
        return adjoint, fd

    def test_free_cube_error_tracks_per_step_rotation(self) -> None:
        adjoint, fd = self._torque_adjoint_and_fd(
            scenes.rigid_free, QuaternionErrorLoss, amplitude=0.3
        )
        rel = np.abs(adjoint - fd) / np.maximum(np.abs(fd), 1e-14)
        # Measured 2.15e-4 at this amplitude (4.3e-4 before external forces were taken
        # out of the finite-difference Hessian-vector products).
        self.assertGreater(float(rel.max()), 1e-4, "approximation gone - update docs")
        self.assertLess(float(rel.max()), 1e-3)

    def test_contact_scene_error_is_small_and_tiny_absolute(self) -> None:
        adjoint, fd = self._torque_adjoint_and_fd(
            lambda: scenes.rigid_on_plane("rich"),
            TranslationErrorLoss,
            amplitude=0.3,
        )
        rel = np.abs(adjoint - fd) / np.maximum(np.abs(fd), 1e-14)
        # Measured 6.5e-4 relative, 3.4e-10 absolute (2.35e-2 / 5.3e-10 before external
        # forces were taken out of the finite-difference Hessian-vector products).
        self.assertGreater(float(rel.max()), 1e-4, "approximation gone - update docs")
        self.assertLess(float(rel.max()), 5e-3)
        self.assertLess(float(np.abs(adjoint - fd).max()), 1e-8)


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

    def _closed_loop(self, history: int, running: bool):
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
        policy = torch.nn.Linear(history * num_dofs, num_dofs).double()
        with torch.no_grad():
            weight = 0.1 * np.cos(np.arange(history * num_dofs * num_dofs, dtype=np.float64)).reshape(
                num_dofs, history * num_dofs
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
            terminal_losses=[terminal],
            step_losses=(lambda step: [running_loss]) if running else None,
        )
        self.addCleanup(rollout.close)
        return scene, chain, policy, rollout, terminal, running_loss, num_steps, history, num_dofs

    def _fd_check(self, history: int, running: bool) -> None:
        scene, chain, policy, rollout, terminal, running_loss, num_steps, history, n = self._closed_loop(
            history, running
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
            for _ in range(num_steps):
                u = weight @ np.concatenate(hist[:history]) + bias
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
