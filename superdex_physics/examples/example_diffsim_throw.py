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

"""Example: Differentiable simulation (throw a cube onto a target)

Optimizes a cube's initial velocity with gradient descent so that, after 0.6 s
of simulation - including impact and sliding on the ground plane - the cube
comes to rest with its center of mass on a target spot. Gradients flow
through every step of the rollout via the per-step adjoint in
``superdex.physics.diffsim``:

  forward:  capture state -> step -> capture state (per step)
  backward: reset_back_propagation; accumulate the terminal loss gradient with
            get_center_of_mass_transform_backward; then, newest step first,
            prepare_back_propagate + back_propagate; finally read
            dL/d(initial velocity) with set_velocity_backward.

Differentiability requires double precision and Backward Euler, so this
example selects double precision before the first physics import. No GUI or
assets are needed.
"""

from __future__ import annotations

import os

# Precision is process-wide and resolved at the first physics import.
os.environ.setdefault("SUPERDEX_PRECISION", "double")

import numpy as np
import superdex.physics as physics

diffsim = physics.diffsim

TIME_STEP = 0.02  # [s]
NUM_STEPS = 30  # 0.6 s rollout
TARGET = np.array([0.9, 0.25, 0.1])  # [m] a spot on the ground
NUM_ITERATIONS = 40
LEARNING_RATE = 2.0

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


def build_scene():
    scene = physics.create_scene("Differentiable Throw")
    scene.set_gravity([0.0, 0.0, -9.81])
    contact = physics.ContactParams(
        penalty_coefficient=1e8, coulomb_friction_coefficient=0.3
    )
    scene.create_rigid_actor(
        name="ground",
        shape=physics.create_plane_shape(normal=[0, 0, 1], distance=0.0),
        is_static=True,
        contact=contact,
    )
    cube = scene.create_rigid_actor(
        name="cube",
        shape=physics.create_tet_mesh_shape(
            coordinates=CUBE_COORDS, connectivity=CUBE_CONN
        ),
        density=1000.0,
        contact=contact,
        world_from_local=physics.TransformRT([0.0, 0.0, 0.5]),
    )
    diffsim.make_scene_differentiable(scene)
    return scene, cube


def rollout_with_gradient(scene, cube, initial_velocity):
    """One forward rollout + adjoint sweep. Returns (loss, dloss/dv0)."""
    cube.set_velocity(initial_velocity, [0.0, 0.0, 0.0])

    pre, post = [], []
    for _ in range(NUM_STEPS):
        pre.append(scene.capture_state())
        scene.step(TIME_STEP)
        post.append(scene.capture_state())

    position = np.asarray(
        cube.get_center_of_mass_transform().translation, dtype=np.float64
    )
    displacement = position - TARGET
    loss = 0.5 * float(displacement @ displacement)

    diffsim.reset_back_propagation(scene)
    diffsim.prepare_back_propagate(scene, post[-1], pre[-1])
    grad_output = np.zeros(7)
    grad_output[:3] = displacement
    diffsim.get_center_of_mass_transform_backward(cube, grad_output)
    for i in range(NUM_STEPS, 0, -1):
        if i != NUM_STEPS:
            diffsim.prepare_back_propagate(scene, post[i - 1], pre[i - 1])
        diffsim.back_propagate(scene)

    grad_linear = np.zeros(3)
    grad_angular = np.zeros(3)
    diffsim.set_velocity_backward(cube, grad_linear, grad_angular)
    # Release only this rollout's states; the caller keeps its own checkpoint.
    for handle in pre + post:
        scene.release_state(handle)
    return loss, grad_linear, position


def main() -> None:
    physics.initialize(num_worker_threads=0)
    scene, cube = build_scene()

    # Solve the adjoint tightly so gradient quality is limited by the model,
    # not by the default 1e-3 adjoint solve tolerance.
    params = diffsim.get_back_propagation_solver_params(scene)
    params.outer_solver_abs_tol = 1e-10
    params.outer_solver_max_iter = 100
    diffsim.set_back_propagation_solver_params(scene, params)

    state_init = scene.capture_state()
    velocity = np.zeros(3)

    print(f"Target: {TARGET}, rollout: {NUM_STEPS} steps x {TIME_STEP} s")
    for iteration in range(NUM_ITERATIONS):
        scene.restore_state(state_init, False)
        loss, gradient, position = rollout_with_gradient(scene, cube, velocity)
        if iteration % 5 == 0 or iteration == NUM_ITERATIONS - 1:
            print(
                f"iter {iteration:3d}  loss {loss:10.6f}  "
                f"final pos [{position[0]:+.3f} {position[1]:+.3f} {position[2]:+.3f}]  "
                f"v0 [{velocity[0]:+.3f} {velocity[1]:+.3f} {velocity[2]:+.3f}]"
            )
        velocity -= LEARNING_RATE * gradient

    print(f"Optimized initial velocity: {velocity}")
    physics.shutdown()


if __name__ == "__main__":
    main()
