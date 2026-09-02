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


def rod_free(num_elements: int = 8, length: float = 0.4):
    """An open elastic rod (no contact, no damping) released under gravity with an
    initial velocity field that bends and twists it: a rigid translation, a
    linearly varying transverse velocity and a twist rate. Exercises the rod
    inertia (incl. the twist inertia), the axial and bend/twist stresses and the
    parallel transport of the material frames between steps. Returns (scene, rod);
    the rod has 4 DoFs per node (3 displacements + twist)."""
    ex = physics.experimental
    scene = physics.create_scene("diffsim_rod_free")
    scene.set_gravity(GRAVITY)
    num_nodes = num_elements + 1
    s = np.linspace(0.0, length, num_nodes)
    # A slightly curved rest polyline (bending is then active from the first step).
    nodes = np.stack([s, 0.03 * np.sin(np.pi * s / length), 0.5 + 0.0 * s], axis=1)
    axes = []
    for e in range(num_elements):
        t = nodes[e + 1] - nodes[e]
        t /= np.linalg.norm(t)
        z = np.array([0.0, 0.0, 1.0])
        a = z - (z @ t) * t
        axes.append(a / np.linalg.norm(a))
    model = ex.generate_tubular_rod_model_data(
        nodes=nodes.tolist(),
        element_frame_axes=[a.tolist() for a in axes],
        radius=0.004,
        num_cross_section_segments=6,
        is_closed_loop=False,
    )
    shape = physics.create_model_shape(model)
    material = ex.RodMaterialParams(
        linear_density=0.05,
        linear_rotational_inertia=2e-6,
        axial_stiffness=2e2,
        torsional_stiffness=2e-2,
        flexural_stiffness=[2e-2, 2e-2],
    )
    rod = ex.create_rod_actor(
        scene,
        ex.RodActorParams(name="rod", shape=shape, material=material, has_gravity=True),
    )
    velocities = np.zeros(4 * num_nodes)
    for i in range(num_nodes):
        velocities[4 * i : 4 * i + 3] = [0.2, 0.0, 0.1 + 0.8 * (s[i] / length)]
        velocities[4 * i + 3] = 3.0 * (1.0 - s[i] / length)  # twist rate [rad/s]
    rod.set_node_velocities_local(velocities)
    return scene, rod


def _rod_actor(
    scene,
    nodes,
    name="rod",
    axis_hint=(0.0, 1.0, 0.0),
    material=None,
    layer="",
    contact=None,
    collider_type=None,
):
    """An open rod actor along ``nodes`` (N x 3) with material frame axes from
    ``axis_hint`` projected orthogonal to each element, 4 DoFs per node."""
    ex = physics.experimental
    nodes = np.asarray(nodes, dtype=np.float64)
    axes = []
    for e in range(len(nodes) - 1):
        t = nodes[e + 1] - nodes[e]
        t /= np.linalg.norm(t)
        a = np.asarray(axis_hint, dtype=np.float64)
        a = a - (a @ t) * t
        axes.append(a / np.linalg.norm(a))
    model = ex.generate_tubular_rod_model_data(
        nodes=nodes.tolist(),
        element_frame_axes=[a.tolist() for a in axes],
        radius=0.004,
        num_cross_section_segments=6,
        is_closed_loop=False,
    )
    if material is None:
        material = ex.RodMaterialParams(
            linear_density=0.05,
            linear_rotational_inertia=2e-6,
            axial_stiffness=2e2,
            torsional_stiffness=2e-2,
            flexural_stiffness=[2e-2, 2e-2],
        )
    params = dict(name=name, shape=physics.create_model_shape(model), material=material, has_gravity=True)
    if layer:
        params["layer"] = layer
    if contact is not None:
        params["contact"] = contact
    if collider_type is not None:
        params["collider_type"] = collider_type
    return ex.create_rod_actor(scene, ex.RodActorParams(**params))


def rod_with_cube(cube_velocity=(0.3, 0.0, 0.0)):
    """A vertical rod pinned at its top node (node position constraint) carrying a
    small rigid cube attached to its bottom node (deformable-node-to-rigid
    constraint): rod-rigid coupling through constraints in one island, no contact.
    The cube starts with ``cube_velocity`` and swings. Returns (scene, rod, cube)."""
    ex = physics.experimental
    scene = physics.create_scene("diffsim_rod_with_cube")
    scene.set_gravity(GRAVITY)
    num_nodes = 7
    z = np.linspace(0.6, 0.3, num_nodes)
    nodes = np.stack([0.0 * z, 0.0 * z, z], axis=1)
    material = ex.RodMaterialParams(
        linear_density=0.05,
        linear_rotational_inertia=2e-6,
        axial_stiffness=2e3,
        torsional_stiffness=2e-1,
        flexural_stiffness=[2e-1, 2e-1],
    )
    rod = _rod_actor(scene, nodes, axis_hint=(1.0, 0.0, 0.0), material=material)
    scene.create_deformable_node_position_constraint(
        actor=rod.get_handle(), node_index=0, position=nodes[0].tolist(), stiffness=2e3
    )
    half = 0.025
    cube = scene.create_rigid_actor(
        name="cube",
        shape=physics.create_tet_mesh_shape(
            coordinates=(np.asarray(CUBE_COORDS) * (half / 0.1)).tolist(), connectivity=CUBE_CONN
        ),
        density=400.0,
        collider_type=physics.ColliderType.NONE,
        world_from_local=physics.TransformRT([0.0, 0.0, 0.3 - half]),
    )
    scene.create_deformable_node_to_rigid_constraint(
        deformable_actor=rod.get_handle(),
        rigid_actor=cube.get_handle(),
        deformable_node_index=num_nodes - 1,
        rigid_local_pos=[0.0, 0.0, half],
        stiffness=2e3,
    )
    cube.set_velocity(list(cube_velocity), [0.0, 0.0, 0.0])
    return scene, rod, cube


def soft_cube_under_rigid(friction: str = "rich", rigid_velocity=(0.2, 0.0, 0.0)):
    """A soft cube on the ground with a dynamic rigid box resting on top of it (1 mm
    into it, like the other contact scenes) and sliding with ``rigid_velocity``:
    sync contact of the soft cube's samples against the box's SDF, in one island.
    The reverse direction (the box's samples against the soft's mapped SDF) is
    disabled by the asymmetric layer filter: deformable colliders have no
    stage-start SDF Hessian, and the backward rejects them. Returns (scene, soft,
    rigid)."""
    scene = physics.create_scene(f"diffsim_soft_under_rigid_{friction}")
    scene.set_gravity(GRAVITY)
    cp = contact_params(friction)
    scene.create_rigid_actor(
        name="ground",
        shape=physics.create_plane_shape(normal=[0, 0, 1], distance=0.0),
        is_static=True,
        contact=cp,
    )
    soft = scene.create_soft_actor(
        name="jelly",
        layer="Soft",
        shape=cube_shape(),
        material=physics.SoftMaterialParams(),
        contact=cp,
        world_from_local=physics.TransformRT([0.0, 0.0, 0.099]),
    )
    rigid = scene.create_rigid_actor(
        name="rigid",
        layer="Rigid",
        shape=cube_shape(),
        density=1000.0,
        contact=cp,
        collider_type=physics.ColliderType.BOX,
        world_from_local=physics.TransformRT([0.0, 0.0, 0.298]),
    )
    scene.enable_layer_contact_asymmetric("Rigid", "Soft", False)
    rigid.set_velocity(list(rigid_velocity), [0.0, 0.0, 0.0])
    return scene, soft, rigid


def rod_onto_cube(
    friction: str = "coulomb",
    ground_friction: str = "none",
    cube_velocity=(0.3, 0.0, 0.0),
    cube_collider=physics.ColliderType.BOX,
    rod_as_collider: bool = False,
):
    """A dynamic cube sliding on the ground with a horizontal rod dropped onto it:
    sync contact of the rod's centerline samples against the cube's SDF (one
    island), with the rod-cube pair in the ``friction`` regime and the cube-ground
    pair in ``ground_friction`` (a pair combines both owners' parameters by
    geometric mean, so "none" on the ground makes that pair frictionless: with
    Coulomb friction on both pairs the cube-velocity gradient was 2.3e-4 off its
    rollout FD, the rod-cube pair alone 2.6e-5 and the ground pair alone 5.5e-5,
    2026-09-02). Rods are no colliders by default; with ``rod_as_collider`` the
    rod gets a point-cloud collider that the cube's samples test against (that
    direction has no adjoint). ``cube_collider`` selects the cube's collider
    type (MESH colliders have no SDF Hessian). Returns (scene, rod, cube)."""
    scene = physics.create_scene(f"diffsim_rod_onto_cube_{friction}")
    scene.set_gravity(GRAVITY)
    cp = contact_params(friction)
    scene.create_rigid_actor(
        name="ground",
        shape=physics.create_plane_shape(normal=[0, 0, 1], distance=0.0),
        is_static=True,
        contact=contact_params(ground_friction),
    )
    cube = scene.create_rigid_actor(
        name="cube",
        layer="Rigid",
        shape=cube_shape(),
        density=1000.0,
        contact=cp,
        collider_type=cube_collider,
        world_from_local=physics.TransformRT([0.15, 0.0, 0.099]),
    )
    x = np.linspace(0.0, 0.3, 7)
    nodes = np.stack([x, np.zeros(7), np.full(7, 0.215)], axis=1)
    rod = _rod_actor(
        scene,
        nodes,
        layer="Rod",
        contact=cp,
        collider_type=physics.ColliderType.POINT_CLOUD if rod_as_collider else None,
    )
    velocities = np.zeros(rod.get_num_dofs())
    velocities[2::4] = -0.5
    rod.set_node_velocities_local(velocities)
    cube.set_velocity(list(cube_velocity), [0.0, 0.0, 0.0])
    return scene, rod, cube


def rod_on_pendulum():
    """A stiff horizontal rod (axial stiffness 2e4 over 5 cm elements, i.e. a
    tendon-like 4e5 N/m per element) pinned at one end (node position
    constraint) and tied at the other to the second link of the controlled
    pendulum of :func:`pendulum` through a node-to-rigid constraint: one island
    with an articulated actor, its pose controller and a rod, no contact.
    Returns (scene, chain, rod)."""
    ex = physics.experimental
    scene, chain = pendulum(with_controller=True)
    num_nodes = 7
    x = np.linspace(0.4, 0.1, num_nodes)
    nodes = np.stack([x, 0.0 * x, 0.75 + 0.0 * x], axis=1)
    material = ex.RodMaterialParams(
        linear_density=0.05,
        linear_rotational_inertia=2e-6,
        axial_stiffness=2e4,
        torsional_stiffness=2e-1,
        flexural_stiffness=[2e-1, 2e-1],
    )
    rod = _rod_actor(scene, nodes, axis_hint=(0.0, 0.0, 1.0), material=material)
    scene.create_deformable_node_position_constraint(
        actor=rod.get_handle(), node_index=0, position=nodes[0].tolist(), stiffness=2e4
    )
    link1 = chain.get_nested_link_actors()[1]
    scene.create_deformable_node_to_rigid_constraint(
        deformable_actor=rod.get_handle(),
        rigid_actor=link1,
        deformable_node_index=num_nodes - 1,
        rigid_local_pos=[0.1, 0.0, 0.0],
        stiffness=2e4,
    )
    # Rod-vs-dynamic-collider contact has no adjoint; the links are dynamic colliders.
    scene.enable_actor_contact_symmetric(
        rod.get_handle(), chain.get_handle(), False, physics.IncludeNestedActors.YES
    )
    return scene, chain, rod


def rod_on_plane(friction: str = "coulomb", height: float = 0.03):
    """A horizontal rod released ``height`` above a static ground plane with a
    downward velocity: centerline contact of the rod against a static collider
    (the one contact case differentiable rods support). Returns (scene, rod)."""
    scene = physics.create_scene(f"diffsim_rod_on_plane_{friction}")
    scene.set_gravity(GRAVITY)
    cp = contact_params(friction)
    scene.create_rigid_actor(
        name="ground",
        shape=physics.create_plane_shape(normal=[0, 0, 1], distance=0.0),
        is_static=True,
        contact=cp,
    )
    num_nodes = 7
    x = np.linspace(0.0, 0.3, num_nodes)
    nodes = np.stack([x, 0.0 * x, height + 0.01 * np.sin(np.pi * x / 0.3)], axis=1)
    rod = _rod_actor(scene, nodes, contact=cp)
    velocities = np.zeros(4 * num_nodes)
    velocities[2::4] = -0.5
    velocities[0::4] = 0.3
    rod.set_node_velocities_local(velocities)
    return scene, rod
