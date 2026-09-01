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

"""Programmatically-built scenes for the diffsim gradient tests.

Equivalents of the internal ``differentiability_test/*.mochi_scene`` assets
(which are not shipped in the open-source export), rebuilt from primitives so
the tests need no asset files. Geometry follows the pybind test suite's
minimal cube tet mesh.
"""

from __future__ import annotations

import numpy as np
import superdex.physics as physics

GRAVITY = [0.0, 0.0, -9.81]

# Minimal cube tet mesh (side length 0.2), as in the pybind test conftest.
CUBE_COORDS = np.array(
    # fmt: off
    [
        -0.1, -0.1, -0.1,
        +0.1, -0.1, -0.1,
        -0.1, +0.1, -0.1,
        +0.1, +0.1, -0.1,
        -0.1, -0.1, +0.1,
        +0.1, -0.1, +0.1,
        -0.1, +0.1, +0.1,
        +0.1, +0.1, +0.1,
    ],
    # fmt: on
    dtype=np.float64,
)
CUBE_CONN = np.array(
    # fmt: off
    [
        0, 1, 2, 4,
        6, 7, 4, 2,
        5, 4, 7, 1,
        3, 2, 1, 7,
        1, 2, 4, 7,
    ],
    # fmt: on
    dtype=np.int32,
)


def cube_shape():
    return physics.create_tet_mesh_shape(
        coordinates=CUBE_COORDS, connectivity=CUBE_CONN
    )


def contact_params(friction: str) -> physics.ContactParams:
    """Contact parameters for one friction regime: none | viscous | coulomb."""
    if friction == "none":
        return physics.ContactParams(penalty_coefficient=1e8)
    if friction == "viscous":
        return physics.ContactParams(
            penalty_coefficient=1e8, viscous_friction_coefficient=0.1
        )
    if friction == "coulomb":
        return physics.ContactParams(
            penalty_coefficient=1e8, coulomb_friction_coefficient=0.4
        )
    if friction == "rich":
        # Every differentiated contact parameter strictly positive, so central
        # finite differences never cross the engine's non-negativity checks.
        return physics.ContactParams(
            penalty_coefficient=1e8,
            coulomb_friction_coefficient=0.4,
            viscous_friction_coefficient=0.1,
            normal_viscous_damping_coefficient=5.0,
        )
    raise ValueError(f"unknown friction regime: {friction}")


def rigid_free():
    """A single free-falling, tumbling cube. Returns (scene, cube)."""
    scene = physics.create_scene("diffsim_rigid_free")
    scene.set_gravity(GRAVITY)
    cube = scene.create_rigid_actor(
        name="cube",
        shape=cube_shape(),
        density=1000.0,
        world_from_local=physics.TransformRT([0.0, 0.0, 1.0]),
    )
    cube.set_velocity([0.3, -0.2, 0.0], [1.0, 0.5, -0.3])
    return scene, cube


def rigid_on_plane(friction: str, initial_velocity=(0.5, 0.0, 0.0)):
    """A cube sliding on a static ground plane. Returns (scene, cube)."""
    scene = physics.create_scene(f"diffsim_rigid_on_plane_{friction}")
    scene.set_gravity(GRAVITY)
    cp = contact_params(friction)
    scene.create_rigid_actor(
        name="ground",
        shape=physics.create_plane_shape(normal=[0, 0, 1], distance=0.0),
        is_static=True,
        contact=cp,
    )
    cube = scene.create_rigid_actor(
        name="cube",
        shape=cube_shape(),
        density=1000.0,
        contact=cp,
        world_from_local=physics.TransformRT([0.0, 0.0, 0.099]),
    )
    cube.set_velocity(list(initial_velocity), [0.0, 0.0, 0.0])
    return scene, cube


def two_cubes_on_plane(friction: str):
    """Two stacked cubes on a plane; the loss target is the bottom cube."""
    scene = physics.create_scene(f"diffsim_two_cubes_{friction}")
    scene.set_gravity(GRAVITY)
    cp = contact_params(friction)
    scene.create_rigid_actor(
        name="ground",
        shape=physics.create_plane_shape(normal=[0, 0, 1], distance=0.0),
        is_static=True,
        contact=cp,
    )
    bottom = scene.create_rigid_actor(
        name="bottom",
        shape=cube_shape(),
        density=1000.0,
        contact=cp,
        world_from_local=physics.TransformRT([0.0, 0.0, 0.099]),
    )
    top = scene.create_rigid_actor(
        name="top",
        shape=cube_shape(),
        density=1000.0,
        contact=cp,
        world_from_local=physics.TransformRT([0.0, 0.0, 0.298]),
    )
    top.set_velocity([0.2, 0.0, 0.0], [0.0, 0.0, 0.0])
    return scene, bottom


def soft_cube(
    mass_damping: float = 0.0,
    initial_velocity=(0.3, 0.0, 0.0),
    squash: float = 0.0,
):
    """A free-floating FEM cube (5-tet mesh). Returns (scene, cube).

    ``squash`` scales the initial z-displacements by ``-squash`` (e.g. 0.1
    compresses the cube by 10% so elastic forces are active from step one);
    ``mass_damping`` sets the material's mass-damping coefficient [1/s].
    Recentering is force-disabled by ``make_scene_differentiable``, so the
    displacements carry the full motion in the fixed local frame.
    """
    scene = physics.create_scene("diffsim_soft_cube")
    scene.set_gravity(GRAVITY)
    material = physics.SoftMaterialParams(mass_damping_coefficient=mass_damping)
    cube = scene.create_soft_actor(
        name="jelly",
        shape=cube_shape(),
        material=material,
        world_from_local=physics.TransformRT([0.0, 0.0, 1.0]),
    )
    num_nodes = cube.get_num_dofs() // 3
    if squash != 0.0:
        rest_z = CUBE_COORDS.reshape(-1, 3)[:, 2]
        displacements = np.zeros(3 * num_nodes)
        displacements[2::3] = -squash * rest_z
        cube.set_displacements(displacements)
    velocities = np.tile(np.asarray(initial_velocity, dtype=np.float64), num_nodes)
    cube.set_node_velocities_local(velocities)
    return scene, cube


def _chain_params() -> tuple[list, list]:
    joints = [
        physics.ArticulatedJointParams(
            name="j0", type=physics.ArticulatedJointType.REVOLUTE, axis=[1, 0, 0]
        ),
        physics.ArticulatedJointParams(
            name="j1",
            type=physics.ArticulatedJointType.REVOLUTE,
            axis=[1, 0, 0],
            parent_link_from_joint=physics.TransformRT([0.0, 0.0, -0.25]),
        ),
    ]
    links = [
        physics.ArticulatedLinkParams(
            name="l0", parent_link=-1, shape=cube_shape(), density=1000.0
        ),
        physics.ArticulatedLinkParams(
            name="l1", parent_link=0, shape=cube_shape(), density=1000.0
        ),
    ]
    return joints, links


def pendulum(with_controller: bool):
    """A fixed-base two-revolute-joint chain. Returns (scene, chain)."""
    scene = physics.create_scene(
        "diffsim_pendulum_controller" if with_controller else "diffsim_pendulum"
    )
    scene.set_gravity(GRAVITY)
    joints, links = _chain_params()
    chain = scene.create_articulated_actor(
        physics.ArticulatedActorParams(
            name="chain",
            joints=joints,
            links=links,
            world_from_root=physics.TransformRT([0.0, 0.0, 1.0]),
        )
    )
    if with_controller:
        tracking = physics.PoseTrackingParams(
            stiffness=50.0, damping=5.0, saturation=-1.0
        )
        chain.add_articulated_pose_controller(
            physics.PoseControllerParams(joint_tracking=[tracking])
        )
    chain.set_articulated_joint_velocities(np.array([0.5, -0.3]))
    return scene, chain


def free_chain_on_plane(friction: str, root_z: float = 0.15):
    """A free-floating root link with one revolute child, falling onto a plane."""
    scene = physics.create_scene(f"diffsim_free_chain_{friction}")
    scene.set_gravity(GRAVITY)
    cp = contact_params(friction)
    scene.create_rigid_actor(
        name="ground",
        shape=physics.create_plane_shape(normal=[0, 0, 1], distance=0.0),
        is_static=True,
        contact=cp,
    )
    joints = [
        physics.ArticulatedJointParams(
            name="root", type=physics.ArticulatedJointType.FREE
        ),
        physics.ArticulatedJointParams(
            name="hinge",
            type=physics.ArticulatedJointType.REVOLUTE,
            axis=[0, 1, 0],
            parent_link_from_joint=physics.TransformRT([0.25, 0.0, 0.0]),
        ),
    ]
    links = [
        physics.ArticulatedLinkParams(
            name="base",
            parent_link=-1,
            shape=cube_shape(),
            density=1000.0,
            contact=cp,
        ),
        physics.ArticulatedLinkParams(
            name="arm",
            parent_link=0,
            shape=cube_shape(),
            density=1000.0,
            contact=cp,
        ),
    ]
    chain = scene.create_articulated_actor(
        physics.ArticulatedActorParams(
            name="free_chain",
            joints=joints,
            links=links,
            world_from_root=physics.TransformRT([0.0, 0.0, root_z]),
        )
    )
    return scene, chain


def soft_cube_on_plane(
    friction: str,
    initial_velocity=(0.3, 0.0, 0.0),
    height: float = 0.099,
    mass_damping: float = 0.0,
):
    """A free FEM cube sliding on a static ground plane. Returns (scene, cube).

    ``friction`` selects the contact regime (see :func:`contact_params`); both
    the plane and the cube carry the same parameters (a contact pair combines
    both owners' values by geometric mean). ``height`` places the cube's
    center; the default rests the bottom face 1 mm into the plane, as in
    :func:`rigid_on_plane`. ``mass_damping`` sets the material's mass-damping
    coefficient [1/s]. Contact against a static collider is *async* contact
    in the engine, the regime the soft contact adjoint covers.
    """
    scene = physics.create_scene(f"diffsim_soft_cube_on_plane_{friction}")
    scene.set_gravity(GRAVITY)
    cp = contact_params(friction)
    scene.create_rigid_actor(
        name="ground",
        shape=physics.create_plane_shape(normal=[0, 0, 1], distance=0.0),
        is_static=True,
        contact=cp,
    )
    cube = scene.create_soft_actor(
        name="jelly",
        shape=cube_shape(),
        material=physics.SoftMaterialParams(mass_damping_coefficient=mass_damping),
        contact=cp,
        world_from_local=physics.TransformRT([0.0, 0.0, height]),
    )
    num_nodes = cube.get_num_dofs() // 3
    velocities = np.tile(np.asarray(initial_velocity, dtype=np.float64), num_nodes)
    cube.set_node_velocities_local(velocities)
    return scene, cube


def chain_pushing_cube(friction: str = "rich"):
    """A fixed-base two-link chain whose lower link rests on the ground beside a
    dynamic cube and, driven by its pose controller, pushes the cube along the
    plane: articulated-vs-dynamic-rigid *sync* contact (same island), the one
    contact combination the other scenes do not exercise. Returns (scene, chain,
    cube); the loss target is the cube."""
    return chain_pushing_cube_with_params(contact_params(friction))


def chain_pushing_cube_with_params(
    cp,
    *,
    chain_cp=None,
    cube_cp=None,
    ground_cp=None,
    cube_collider: bool = True,
    link_collider: bool = True,
):
    """:func:`chain_pushing_cube` with explicit contact parameters: ``cp`` for every
    actor unless overridden per actor (a contact pair combines both owners'
    parameters by geometric mean, so a zero coefficient on one owner makes the
    pair frictionless). ``cube_collider`` / ``link_collider`` select which body
    owns an SDF collider, i.e. whose surface the other body's samples are tested
    against (both by default)."""
    chain_cp = cp if chain_cp is None else chain_cp
    cube_cp = cp if cube_cp is None else cube_cp
    ground_cp = cp if ground_cp is None else ground_cp
    cube_collider_type = (
        physics.ColliderType.SDF if cube_collider else physics.ColliderType.NONE
    )
    link_collider_type = (
        physics.ColliderType.SDF if link_collider else physics.ColliderType.NONE
    )
    scene = physics.create_scene("diffsim_chain_pushing_cube")
    scene.set_gravity(GRAVITY)
    scene.create_rigid_actor(
        name="ground",
        shape=physics.create_plane_shape(normal=[0, 0, 1], distance=0.0),
        is_static=True,
        contact=ground_cp,
    )
    cube = scene.create_rigid_actor(
        name="cube",
        shape=cube_shape(),
        density=500.0,
        contact=cube_cp,
        world_from_local=physics.TransformRT([0.25, 0.0, 0.099]),
        collider_type=cube_collider_type,
    )
    joints = [
        physics.ArticulatedJointParams(
            name="j0", type=physics.ArticulatedJointType.REVOLUTE, axis=[0, 1, 0]
        ),
        physics.ArticulatedJointParams(
            name="j1",
            type=physics.ArticulatedJointType.REVOLUTE,
            axis=[0, 1, 0],
            parent_link_from_joint=physics.TransformRT([0.0, 0.0, -0.25]),
        ),
    ]
    links = [
        physics.ArticulatedLinkParams(
            name="l0",
            parent_link=-1,
            shape=cube_shape(),
            density=1000.0,
            contact=chain_cp,
            collider_type=link_collider_type,
        ),
        physics.ArticulatedLinkParams(
            name="l1",
            parent_link=0,
            shape=cube_shape(),
            density=1000.0,
            contact=chain_cp,
            collider_type=link_collider_type,
        ),
    ]
    chain = scene.create_articulated_actor(
        physics.ArticulatedActorParams(
            name="chain",
            joints=joints,
            links=links,
            world_from_root=physics.TransformRT([0.0, 0.0, 0.352]),
        )
    )
    chain.add_articulated_pose_controller(
        physics.PoseControllerParams(
            joint_tracking=[
                physics.PoseTrackingParams(stiffness=80.0, damping=8.0, saturation=-1.0)
            ]
        )
    )
    return scene, chain, cube
