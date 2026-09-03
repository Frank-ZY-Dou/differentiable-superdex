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

"""Example: System identification with parameter adjoints

Recovers a cube's Coulomb friction coefficient and density from an observed
sliding trajectory, using the engine's parameter gradients
(``diffsim.set_contact_params_backward`` and ``diffsim.set_density_backward``)
together with the multi-step rollout driver
(:class:`superdex.physics.diffsim_rollout.DifferentiableRollout`).

Identifiability needs both parameters to shape the motion independently: a
known horizontal force in the first phase makes the acceleration depend on
the mass (``F / m``), while the force-free second phase is a pure friction
slide whose deceleration (``mu g``) is mass-independent. The loss is a
running (per-step) tracking cost against the observed center-of-mass
positions, so every step contributes gradient information.

Three modeling points this example is careful about (each one, when violated,
produced a biased or runaway identification during development):

1. Matched forward model: ``make_scene_differentiable`` switches the forward
   physics (explicit contact normals, no Newton-Euler inertia forces), so the
   observations are recorded with the *same* differentiable configuration the
   estimator simulates. Otherwise the loss at the true parameters is not zero
   and the estimate is biased.
2. Pair materials: a contact pair uses the *geometric mean* of the two
   owners' contact parameters. The ground material is treated as known and
   fixed (a calibrated rig), and only the cube-side coefficient is estimated;
   its gradient is read from the cube's accumulator alone.
3. Forward solver tolerance bounds gradient fidelity: with the default Newton
   tolerance the trajectory satisfies the step equations only to ~1e-3, which
   showed up as 4-16 percent friction-gradient error against finite
   differences. With ``abs_tol = 1e-12`` the adjoint matches rollout finite
   differences to ~1e-6 relative.

``--mode soft`` identifies instead the Young's modulus and Poisson's ratio of a
soft (neo-Hookean FEM) cube dropped onto the ground, from the observed nodal
displacement fields of the impact and the rebound: the squash depth follows
the modulus, the lateral bulge the Poisson's ratio. The gradients come from
``diffsim.set_soft_material_params_backward`` (finite differences of the
assembled residual under perturbed material parameters, the per-element Lame
constants rebuilt from both fields).

No GUI or assets are needed. Double precision is selected before the first
physics import (differentiability requirement).
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("SUPERDEX_PRECISION", "double")

import numpy as np
import superdex.physics as physics
from superdex.physics.diffsim_rollout import DifferentiableRollout

diffsim = physics.diffsim

TIME_STEP = 0.01
NUM_STEPS = 30
# Two-phase excitation for identifiability: a strong known force in the first
# half makes the acceleration mass-dependent (F/m), the force-free second half
# is a pure friction slide whose deceleration (mu*g) is mass-independent.
FORCE_STEPS = 15
APPLIED_FORCE = np.array([40.0, 0.0, 0.0])  # [N] known excitation
TRUE_FRICTION = 0.4
TRUE_DENSITY = 1000.0  # [kg/m^3]
INITIAL_GUESS = {"friction": 0.15, "density": 1600.0}
NUM_ITERATIONS = 400
ADAM_LR = 0.05  # decayed x0.3 at iteration 250
MAX_RECOVERY_ERROR = 0.015  # hard acceptance threshold on both parameters

# Minimal cube tet mesh (side length 0.2).
# fmt: off
CUBE_COORDS = np.array([
    -0.1, -0.1, -0.1,  +0.1, -0.1, -0.1,  -0.1, +0.1, -0.1,  +0.1, +0.1, -0.1,
    -0.1, -0.1, +0.1,  +0.1, -0.1, +0.1,  -0.1, +0.1, +0.1,  +0.1, +0.1, +0.1,
], dtype=np.float64)
CUBE_CONN = np.array([
    0, 1, 2, 4,  6, 7, 4, 2,  5, 4, 7, 1,  3, 2, 1, 7,  1, 2, 4, 7,
], dtype=np.int32)
# fmt: on

FORCE_DOFS = np.array([0, 1, 2], dtype=np.int32)


GROUND_FRICTION = 0.5  # known rig material (pair value = geometric mean)

# Soft mode: a 3x3x3-cell neo-Hookean cube (side 0.2 m) dropped onto the ground.
SOFT_CELLS = 3
SOFT_SIDE = 0.2
SOFT_TRUE = {"young": 4.0e4, "poisson": 0.45}
SOFT_GUESS = {"young": 1.0e5, "poisson": 0.30}
SOFT_DENSITY = 1000.0  # [kg/m^3], known
SOFT_MASS_DAMPING = 1.0  # [1/s], known
SOFT_DROP_VELOCITY = -1.0  # [m/s] at release, 1 cm above the ground
SOFT_NUM_STEPS = 40
SOFT_NUM_ITERATIONS = 150
SOFT_ADAM_LR = 0.05
SOFT_POISSON_BOUNDS = (0.05, 0.49)


def build_scene(friction: float, density: float):
    """Estimator/ground-truth scene under the differentiable forward model."""
    scene = physics.create_scene("SysID")
    scene.set_gravity([0.0, 0.0, -9.81])
    scene.create_rigid_actor(
        name="ground",
        shape=physics.create_plane_shape(normal=[0, 0, 1], distance=0.0),
        is_static=True,
        contact=physics.ContactParams(
            penalty_coefficient=1e8, coulomb_friction_coefficient=GROUND_FRICTION
        ),
    )
    cube = scene.create_rigid_actor(
        name="cube",
        shape=physics.create_tet_mesh_shape(
            coordinates=CUBE_COORDS, connectivity=CUBE_CONN
        ),
        density=density,
        contact=physics.ContactParams(
            penalty_coefficient=1e8, coulomb_friction_coefficient=friction
        ),
        world_from_local=physics.TransformRT([0.0, 0.0, 0.099]),
    )
    cube.set_velocity([0.8, 0.0, 0.0], [0.0, 0.0, 0.0])

    # Same forward model for data generation and estimation, and a tight
    # Newton solve so the trajectory satisfies the step equations to well
    # below the finite-difference/adjoint agreement we rely on.
    diffsim.make_scene_differentiable(scene)
    solver = scene.get_solver_params()
    newton = solver.non_linear_solver
    newton.max_iter = 200
    newton.abs_tol = 1e-12
    newton.rel_tol = 1e-12
    solver.non_linear_solver = newton
    scene.set_solver_params(solver)
    return scene, cube


def box_tet_mesh(size: float, cells: int):
    """A cube of ``cells^3`` hexahedra, each split into 5 tetrahedra (alternating
    orientation so the faces match); returns (coordinates, connectivity)."""
    n = cells + 1
    grid = np.linspace(-0.5 * size, 0.5 * size, n)
    coordinates = np.array([[x, y, z] for z in grid for y in grid for x in grid])
    index = lambda i, j, k: (k * n + j) * n + i
    tets = []
    for k in range(cells):
        for j in range(cells):
            for i in range(cells):
                c = [index(i + a, j + b, k + d) for d in (0, 1) for b in (0, 1) for a in (0, 1)]
                if (i + j + k) % 2 == 0:
                    local = [(0, 1, 2, 4), (1, 3, 2, 7), (1, 4, 5, 7), (2, 4, 6, 7), (1, 2, 4, 7)]
                else:
                    local = [(0, 1, 3, 5), (0, 2, 3, 6), (0, 4, 5, 6), (3, 5, 6, 7), (0, 3, 5, 6)]
                tets.extend([c[a] for a in tet] for tet in local)
    return coordinates.reshape(-1), np.array(tets, dtype=np.int32).reshape(-1)


def build_soft_scene(young: float, poisson: float):
    """Ground-truth / estimator scene of the soft mode: the cube is released 1 cm
    above the ground, falling at SOFT_DROP_VELOCITY."""
    scene = physics.create_scene("SysID soft")
    scene.set_gravity([0.0, 0.0, -9.81])
    contact = physics.ContactParams(penalty_coefficient=1e7, coulomb_friction_coefficient=0.4)
    scene.create_rigid_actor(
        name="ground",
        shape=physics.create_plane_shape(normal=[0, 0, 1], distance=0.0),
        is_static=True,
        contact=contact,
    )
    coordinates, connectivity = box_tet_mesh(SOFT_SIDE, SOFT_CELLS)
    material = physics.SoftMaterialParams(
        density=SOFT_DENSITY, mass_damping_coefficient=SOFT_MASS_DAMPING
    )
    material.neo_hookean = physics.NeoHookeanMaterialParams(
        youngs_modulus=young, poisson_ratio=poisson
    )
    cube = scene.create_soft_actor(
        name="cube",
        shape=physics.create_tet_mesh_shape(coordinates=coordinates, connectivity=connectivity),
        material=material,
        contact=contact,
        world_from_local=physics.TransformRT([0.0, 0.0, 0.5 * SOFT_SIDE + 0.01]),
    )
    num_nodes = coordinates.size // 3
    cube.set_node_velocities_local(np.tile([0.0, 0.0, SOFT_DROP_VELOCITY], num_nodes))
    diffsim.make_scene_differentiable(scene)  # also disables recentering (fixed root frame)
    solver = scene.get_solver_params()
    newton = solver.non_linear_solver
    newton.max_iter = 200
    newton.abs_tol = 1e-10
    newton.rel_tol = 1e-10
    solver.non_linear_solver = newton
    scene.set_solver_params(solver)
    return scene, cube


def record_soft_observations() -> np.ndarray:
    """Simulate the ground-truth soft cube and record its displacement field per step."""
    scene, cube = build_soft_scene(SOFT_TRUE["young"], SOFT_TRUE["poisson"])
    observations = []
    for _ in range(SOFT_NUM_STEPS):
        scene.step(TIME_STEP)
        observations.append(np.asarray(cube.get_displacements(), dtype=np.float64).copy())
    physics.destroy_scene(scene)
    return np.stack(observations)


class DisplacementTrackingLoss:
    """0.5 * || u_step - observation_step ||^2 on the nodal displacements, one per step."""

    def __init__(self, cube, observation: np.ndarray):
        self.cube = cube
        self.observation = observation

    def _residual(self) -> np.ndarray:
        return np.asarray(self.cube.get_displacements(), dtype=np.float64) - self.observation

    def value(self) -> float:
        residual = self._residual()
        return 0.5 * float(residual @ residual)

    def accumulate_output_grad(self) -> None:
        diffsim.get_displacements_backward(self.cube, self._residual())


def main_soft() -> None:
    observations = record_soft_observations()
    scene, cube = build_soft_scene(SOFT_GUESS["young"], SOFT_GUESS["poisson"])
    params = diffsim.get_back_propagation_solver_params(scene)
    params.outer_solver_max_iter = 100
    diffsim.set_back_propagation_solver_params(scene, params)
    state_init = scene.capture_state()
    losses = [DisplacementTrackingLoss(cube, observations[step]) for step in range(SOFT_NUM_STEPS)]
    rollout = DifferentiableRollout(scene, dt=TIME_STEP, num_steps=SOFT_NUM_STEPS)
    # Unconstrained coordinates: log for the positive modulus, plain (clipped to the
    # physical range) for the Poisson's ratio.
    theta = np.array([np.log(SOFT_GUESS["young"]), SOFT_GUESS["poisson"]])
    adam_m = np.zeros(2)
    adam_v = np.zeros(2)
    print(
        f"true: E={SOFT_TRUE['young']:.0f}, nu={SOFT_TRUE['poisson']}  |  "
        f"guess: E={SOFT_GUESS['young']:.0f}, nu={SOFT_GUESS['poisson']}"
    )
    for iteration in range(SOFT_NUM_ITERATIONS):
        young, poisson = float(np.exp(theta[0])), float(theta[1])
        scene.restore_state(state_init, False)
        material = cube.get_soft_material_params()
        material.neo_hookean = physics.NeoHookeanMaterialParams(
            youngs_modulus=young, poisson_ratio=poisson
        )
        cube.set_soft_material_params(material)
        result = rollout.run(step_losses=lambda step: [losses[step]])
        material_grad = np.zeros(4)  # young, poisson, density, mass damping
        diffsim.set_soft_material_params_backward(cube, material_grad)
        gradient = np.array([material_grad[0] * young, material_grad[1]])  # d/d(ln E), d/d(nu)
        if iteration % 10 == 0 or iteration == SOFT_NUM_ITERATIONS - 1:
            print(
                f"iter {iteration:3d}  loss {result.loss:11.3e}  E {young:9.1f}  nu {poisson:6.4f}  "
                f"dL/dlnE {gradient[0]:+.3e}  dL/dnu {gradient[1]:+.3e}"
            )
        adam_m = 0.9 * adam_m + 0.1 * gradient
        adam_v = 0.999 * adam_v + 0.001 * gradient * gradient
        m_hat = adam_m / (1.0 - 0.9 ** (iteration + 1))
        v_hat = adam_v / (1.0 - 0.999 ** (iteration + 1))
        theta -= SOFT_ADAM_LR * m_hat / (np.sqrt(v_hat) + 1e-12)
        theta[1] = float(np.clip(theta[1], *SOFT_POISSON_BOUNDS))
    young, poisson = float(np.exp(theta[0])), float(theta[1])
    e_error = abs(young - SOFT_TRUE["young"]) / SOFT_TRUE["young"]
    nu_error = abs(poisson - SOFT_TRUE["poisson"]) / SOFT_TRUE["poisson"]
    print(
        f"recovered: E={young:.1f} ({100 * e_error:.2f}% err), "
        f"nu={poisson:.4f} ({100 * nu_error:.2f}% err)"
    )
    scene.release_all_states()
    physics.destroy_scene(scene)
    if e_error > MAX_RECOVERY_ERROR or nu_error > MAX_RECOVERY_ERROR:
        raise RuntimeError(
            "system identification did not recover the true material: "
            f"E error {100 * e_error:.2f}%, nu error {100 * nu_error:.2f}% "
            f"(threshold {100 * MAX_RECOVERY_ERROR:.1f}%)"
        )


def apply_inputs_factory(cube):
    def apply_inputs(step: int) -> None:
        force = APPLIED_FORCE if step < FORCE_STEPS else np.zeros(3)
        cube.set_external_forces_on_dofs(FORCE_DOFS, force)

    return apply_inputs


def record_observations() -> np.ndarray:
    """Simulate the ground-truth system and record com positions per step."""
    scene, cube = build_scene(TRUE_FRICTION, TRUE_DENSITY)
    apply_inputs = apply_inputs_factory(cube)
    observations = np.zeros((NUM_STEPS, 3))
    for step in range(NUM_STEPS):
        apply_inputs(step)
        scene.step(TIME_STEP)
        observations[step] = np.asarray(
            cube.get_center_of_mass_transform().translation, dtype=np.float64
        )
    physics.destroy_scene(scene)
    return observations


class TrackingLoss:
    """0.5 * || p_step - observation_step ||^2, one instance per step."""

    def __init__(self, cube, observation: np.ndarray):
        self.cube = cube
        self.observation = observation

    def _displacement(self) -> np.ndarray:
        position = np.asarray(
            self.cube.get_center_of_mass_transform().translation, dtype=np.float64
        )
        return position - self.observation

    def value(self) -> float:
        displacement = self._displacement()
        return 0.5 * float(displacement @ displacement)

    def accumulate_output_grad(self) -> None:
        grad = np.zeros(7)
        grad[:3] = self._displacement()
        diffsim.get_center_of_mass_transform_backward(self.cube, grad)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--mode", choices=["rigid", "soft"], default="rigid")
    args = parser.parse_args()
    physics.initialize(num_worker_threads=0)
    if args.mode == "soft":
        main_soft()
        physics.shutdown()
        return
    observations = record_observations()

    # Estimator scene starts from the wrong parameters.
    scene, cube = build_scene(INITIAL_GUESS["friction"], INITIAL_GUESS["density"])
    params = diffsim.get_back_propagation_solver_params(scene)
    params.outer_solver_abs_tol = 1e-10
    params.outer_solver_max_iter = 100
    diffsim.set_back_propagation_solver_params(scene, params)

    apply_inputs = apply_inputs_factory(cube)
    state_init = scene.capture_state()
    losses = [TrackingLoss(cube, observations[step]) for step in range(NUM_STEPS)]
    rollout = DifferentiableRollout(scene, dt=TIME_STEP, num_steps=NUM_STEPS)

    # Parameters in unconstrained coordinates: log for the positive density,
    # plain for friction (projected at zero). A small Adam keeps the two very
    # differently scaled gradients moving at comparable rates.
    theta = np.array([INITIAL_GUESS["friction"], np.log(INITIAL_GUESS["density"])])
    adam_m = np.zeros(2)
    adam_v = np.zeros(2)

    print(
        f"true: mu={TRUE_FRICTION}, rho={TRUE_DENSITY}  |  "
        f"guess: mu={theta[0]}, rho={np.exp(theta[1]):.1f}"
    )
    for iteration in range(NUM_ITERATIONS):
        friction, log_density = float(theta[0]), float(theta[1])
        scene.restore_state(state_init, False)
        contact = cube.get_contact_params()
        contact.coulomb_friction_coefficient = friction
        cube.set_contact_params(contact)
        cube.set_density(float(np.exp(log_density)))

        result = rollout.run(
            apply_inputs=apply_inputs,
            step_losses=lambda step: [losses[step]],
        )
        contact_grad = np.zeros(4)
        diffsim.set_contact_params_backward(cube, contact_grad)
        density_grad = np.zeros(1)
        diffsim.set_density_backward(cube, density_grad)
        gradient = np.array(
            [
                contact_grad[1],  # coulomb_friction_coefficient slot
                density_grad[0] * float(np.exp(log_density)),  # d/d(log rho)
            ]
        )

        if iteration % 20 == 0 or iteration == NUM_ITERATIONS - 1:
            print(
                f"iter {iteration:3d}  loss {result.loss:11.3e}  "
                f"mu {friction:7.4f}  rho {np.exp(log_density):8.2f}  "
                f"dL/dmu {gradient[0]:+.3e}  dL/dlnrho {gradient[1]:+.3e}"
            )

        adam_m = 0.9 * adam_m + 0.1 * gradient
        adam_v = 0.999 * adam_v + 0.001 * gradient * gradient
        m_hat = adam_m / (1.0 - 0.9 ** (iteration + 1))
        v_hat = adam_v / (1.0 - 0.999 ** (iteration + 1))
        lr = ADAM_LR * (0.3 if iteration >= 250 else 1.0)
        theta -= lr * m_hat / (np.sqrt(v_hat) + 1e-12)
        theta[0] = max(0.0, theta[0])

    friction, density = float(theta[0]), float(np.exp(theta[1]))
    mu_error = abs(friction - TRUE_FRICTION) / TRUE_FRICTION
    rho_error = abs(density - TRUE_DENSITY) / TRUE_DENSITY
    print(
        f"recovered: mu={friction:.4f} ({100 * mu_error:.2f}% err), "
        f"rho={density:.2f} ({100 * rho_error:.2f}% err)"
    )
    physics.shutdown()
    if mu_error > MAX_RECOVERY_ERROR or rho_error > MAX_RECOVERY_ERROR:
        raise RuntimeError(
            "system identification did not recover the true parameters: "
            f"mu error {100 * mu_error:.2f}%, rho error {100 * rho_error:.2f}% "
            f"(threshold {100 * MAX_RECOVERY_ERROR:.1f}%)"
        )


if __name__ == "__main__":
    main()
