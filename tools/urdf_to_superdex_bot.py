#!/usr/bin/env python3
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

"""Convert a URDF robot description into a SuperDex bot package.

The kinematics, inertias and joint limits come from SuperDex's own URDF importer
(``superdex.robotics.load_bot_prefab_from_urdf_file``); the meshes referenced by the
URDF are written next to the bot file: collision meshes as binary STL (the engine bakes
their signed distance fields when the bot is created) and visual meshes as GLB in the
Y-up convention of the shipped assets, colored with the URDF material. A collision mesh
is kept when it is a closed volume with at most MAX_COLLISION_FACES faces (faces without
area dropped); otherwise its convex hull replaces it (an open mesh has no inside for a
distance field). Hulls are subdivided until their mean edge is below COLLISION_MAX_EDGE,
which sets the engine's voxel size.
Contact between the root link and the links not attached to it is disabled, as in the
shipped hand assets. Mesh scales of the URDF are baked into the written meshes.

A link without a collision mesh gets no shape, and the engine gives a shapeless link no
mass: the inertial of such a link is folded into its parent when the joint between them is
fixed (parallel-axis update in the parent's frame), and the link stays a massless frame;
a shapeless link on a moving joint is reported and left massless. Joint dynamics carried
by the URDF (viscous damping and Coulomb friction, joint inertia, the importer's limit
stiffness and damping) are written to the package. A root that is a bare ``world`` frame
(no inertial, no mesh, the ROS convention for a robot welded to the world) makes the world
joint Hard (fixed base); any other root stays floating unless ``--base fixed`` says
otherwise, and ``--base floating`` keeps a Free root. The importer drops the rotation of an
inertial origin, which the tool reports.
Mimic joints are not imported by SuperDex: their dofs are independent in the package.

    python tools/urdf_to_superdex_bot.py <urdf> <out_dir> --name <bot_name> [--mesh-dir DIR]
        [--default-pose JOINT=VALUE ...] [--base auto|fixed|floating]

Run it with the superdex environment of this repository (it needs superdex.physics,
superdex.robotics, numpy and trimesh).

The package is verified after writing: it is loaded back, a bot is created from it and
from the URDF prefab, and every link transform of the two is compared at a pose that
moves every joint inside its limits.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import xml.etree.ElementTree as ET

os.environ.setdefault("SUPERDEX_PRECISION", "double")

import numpy as np
import superdex.physics as physics
import superdex.robotics as robotics
import trimesh

# The viewer and the Blender export map a GLB vertex (x, y, z) to the shape frame as
# (x, -z, y); a shape-frame vertex is therefore stored as (x, z, -y).
TO_GLB = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
MAX_COLLISION_FACES = 3000
# Target mean edge of a hull [m], a tenth of the link's extent within these bounds: the engine's
# voxel size is a quarter of the mean edge (2 mm on a 10 cm palm, 0.75 mm on a fingertip).
COLLISION_EDGE_FRACTION, COLLISION_EDGE_MIN, COLLISION_EDGE_MAX = 0.1, 0.003, 0.008


def load_scaled(source: pathlib.Path, scale) -> trimesh.Trimesh:
    """The mesh with the URDF's per-axis scale baked into its vertices."""
    mesh = trimesh.load(str(source), force="mesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * np.asarray(scale, dtype=np.float64)
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(mesh.faces), process=False)


def write_collision(source: pathlib.Path, target: pathlib.Path, scale) -> str:
    """Writes the collision mesh (or its convex hull) as binary STL; returns a note."""
    mesh = load_scaled(source, scale)
    # A face without area has no normal and the engine's volume of the shape is not finite.
    # Slivers with an area are left alone: some vendors' meshes are closed only through them.
    mesh.update_faces(mesh.area_faces > 0.0)
    if mesh.is_volume and len(mesh.faces) <= MAX_COLLISION_FACES:
        out = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
        note = f"{len(mesh.faces)} faces kept"
    else:
        # The hull of the vertices rounded to 0.1 mm: qhull's triangulation of the raw
        # vertices of these scanned parts is not always closed.
        points = np.unique(np.round(np.asarray(mesh.vertices, dtype=np.float64), 4), axis=0)
        out = trimesh.Trimesh(vertices=points, process=False).convex_hull
        extent = float(np.max(out.bounds[1] - out.bounds[0]))
        max_edge = float(np.clip(COLLISION_EDGE_FRACTION * extent, COLLISION_EDGE_MIN, COLLISION_EDGE_MAX))
        # Uniform subdivision keeps the hull closed (every edge is split on both sides);
        # each level halves the edge lengths.
        while float(out.edges_unique_length.mean()) > max_edge and len(out.faces) < 20000:
            out = out.subdivide()
        out = trimesh.Trimesh(vertices=out.vertices, faces=out.faces, process=True)  # merge split vertices
        if not out.is_watertight:
            raise RuntimeError(f"the convex hull of {source} is not watertight after subdivision")
        why = "open" if not mesh.is_volume else f"{len(mesh.faces)} faces"
        note = f"{why} -> convex hull, {len(out.faces)} faces"
    target.parent.mkdir(parents=True, exist_ok=True)
    out.export(str(target))
    return note


def parse_urdf_meshes(urdf: pathlib.Path) -> dict[str, dict]:
    """Per link: the first visual and collision mesh filenames and the visual color."""
    root = ET.parse(urdf).getroot()
    materials = {}
    for material in root.findall("material"):
        color = material.find("color")
        if material.get("name") and color is not None:
            materials[material.get("name")] = [float(c) for c in color.get("rgba").split()]
    links = {}
    for link in root.findall("link"):
        entry = {"visual": None, "collision": None, "color": None}
        for tag in ("visual", "collision"):
            for element in link.findall(tag):
                mesh = element.find("geometry/mesh")
                if mesh is None or entry[tag] is not None:
                    continue
                entry[tag] = mesh.get("filename")
                if tag == "visual":
                    material = element.find("material")
                    if material is not None:
                        color = material.find("color")
                        if color is not None:
                            entry["color"] = [float(c) for c in color.get("rgba").split()]
                        elif material.get("name") in materials:
                            entry["color"] = materials[material.get("name")]
        links[link.get("name")] = entry
    return links


def resolve_mesh(filename: str, urdf_dir: pathlib.Path, mesh_dir: pathlib.Path | None) -> pathlib.Path:
    name = filename
    if name.startswith("package://"):
        name = name.split("/", 3)[-1]  # drop package://<package>/
    candidates = [urdf_dir / name, urdf_dir / pathlib.Path(name).name]
    if mesh_dir is not None:
        candidates = [mesh_dir / name, mesh_dir / pathlib.Path(name).name] + candidates
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"mesh {filename!r} of the URDF not found (looked at {candidates})")


def vec(values) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=np.float64).reshape(-1)]


def is_identity(rotation, translation, scale=None) -> bool:
    q = np.asarray(rotation, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64)
    ok = np.allclose(q, [0, 0, 0, 1], atol=1e-12) and np.allclose(t, 0, atol=1e-12)
    if scale is not None:
        ok = ok and np.allclose(np.asarray(scale, dtype=np.float64), 1, atol=1e-12)
    return ok


def write_glb(source: pathlib.Path, target: pathlib.Path, color, scale) -> np.ndarray:
    mesh = load_scaled(source, scale)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    bounds = np.stack([vertices.min(0), vertices.max(0)])
    out = trimesh.Trimesh(vertices=vertices @ TO_GLB.T, faces=np.asarray(mesh.faces), process=False)
    rgba = [0.8, 0.8, 0.8, 1.0] if color is None else list(color)
    out.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[int(round(255 * c)) for c in rgba], metallicFactor=0.0, roughnessFactor=0.6
        )
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    out.export(str(target))
    return bounds


def quat_to_matrix(q) -> np.ndarray:
    """Rotation matrix of a unit quaternion (x, y, z, w)."""
    x, y, z, w = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def transform_matrix(transform) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) of a TransformRT."""
    return quat_to_matrix(transform.rotation.tolist()), np.asarray(transform.translation, dtype=np.float64)


def inertia_matrix(moment_of_inertia) -> np.ndarray:
    xx, xy, xz, yy, yz, zz = np.asarray(moment_of_inertia, dtype=np.float64).reshape(6)
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def inertia_vector(matrix) -> list[float]:
    m = np.asarray(matrix, dtype=np.float64)
    return [float(m[0, 0]), float(m[0, 1]), float(m[0, 2]), float(m[1, 1]), float(m[1, 2]), float(m[2, 2])]


def shifted_inertia(inertia, mass, offset) -> np.ndarray:
    """Inertia about a point ``offset`` away from the center of mass (parallel axis)."""
    d = np.asarray(offset, dtype=np.float64)
    return inertia + mass * (float(d @ d) * np.eye(3) - np.outer(d, d))


def fold_shapeless_links(prefab, meshes) -> None:
    """Folds the inertial of every link without a collision mesh into its parent across a
    fixed joint; the engine gives a shapeless link no mass, so this keeps the robot's mass
    properties. Children are handled before their parents (links index their parents by a
    smaller index), so a chain of frames folds all the way to the first link with a shape."""
    for i in reversed(range(len(prefab.links))):
        link = prefab.links[i]
        if link.mass is None or float(link.mass) <= 0.0:
            continue
        refs = meshes.get(link.name, {"visual": None, "collision": None, "color": None})
        if refs["collision"] is not None:
            continue
        joint = prefab.joints[i]
        if link.parent_link < 0 or joint.type != physics.ArticulatedJointType.HARD:
            print(
                f"  {link.name}: no collision mesh and a moving joint ({joint_type_name(joint.type)}); "
                f"the engine will treat it as massless ({float(link.mass):.4g} kg lost)"
            )
            continue
        parent = prefab.links[link.parent_link]
        # child link frame in the parent link frame: parent_link_from_joint * parent_joint_from_link
        r_joint, t_joint = transform_matrix(joint.parent_link_from_joint)
        r_link, t_link = transform_matrix(link.parent_joint_from_link)
        rotation = r_joint @ r_link
        translation = t_joint + r_joint @ t_link
        mass_c = float(link.mass)
        com_c = translation + rotation @ np.asarray(link.center_of_mass, dtype=np.float64)
        inertia_c = rotation @ inertia_matrix(link.moment_of_inertia) @ rotation.T
        if parent.mass is None or float(parent.mass) <= 0.0:
            mass_p, com_p, inertia_p = 0.0, np.zeros(3), np.zeros((3, 3))
        else:
            mass_p = float(parent.mass)
            com_p = np.asarray(parent.center_of_mass, dtype=np.float64)
            inertia_p = inertia_matrix(parent.moment_of_inertia)
        mass = mass_p + mass_c
        com = (mass_p * com_p + mass_c * com_c) / mass
        inertia = shifted_inertia(inertia_p, mass_p, com_p - com)
        inertia = inertia + shifted_inertia(inertia_c, mass_c, com_c - com)
        parent.mass = mass
        parent.center_of_mass = com.tolist()
        parent.moment_of_inertia = inertia_vector(inertia)
        prefab.links[link.parent_link] = parent
        link.mass = None
        prefab.links[i] = link
        print(f"  {link.name}: no collision mesh, {mass_c:.4g} kg folded into {parent.name} (fixed joint)")


def urdf_facts(urdf: pathlib.Path, prefab, meshes) -> dict:
    """Whether the base is fixed, the mimic joints, and the links whose inertial frame is
    rotated. The importer's root is the one link that is no joint's child; the base is fixed
    when that root is a bare ``world`` frame (no inertial, no mesh), the ROS convention for a
    robot welded to the world. A bare root under another name (``base_link``,
    ``base_footprint``) stays floating, as does a root that carries a body."""
    root = ET.parse(urdf).getroot()
    root_link = prefab.links[0]
    refs = meshes.get(root_link.name, {"visual": None, "collision": None, "color": None})
    bare_root = (
        root_link.name == "world"
        and root_link.mass is None
        and refs["visual"] is None
        and refs["collision"] is None
    )
    mimics = [joint.get("name") for joint in root.findall("joint") if joint.find("mimic") is not None]
    rotated_inertials = []
    for link in root.findall("link"):
        origin = link.find("inertial/origin")
        if origin is not None and any(abs(float(v)) > 0.0 for v in (origin.get("rpy") or "0 0 0").split()):
            rotated_inertials.append(link.get("name"))
    return {"fixed_base": bare_root, "mimics": mimics, "rotated_inertials": rotated_inertials}


def joint_type_name(joint_type) -> str:
    return {
        physics.ArticulatedJointType.FREE: "Free",
        physics.ArticulatedJointType.HARD: "Hard",
        physics.ArticulatedJointType.REVOLUTE: "Revolute",
        physics.ArticulatedJointType.PRISMATIC: "Prismatic",
    }[joint_type]


def convert(
    urdf: pathlib.Path,
    out_dir: pathlib.Path,
    name: str,
    mesh_dir: pathlib.Path | None,
    default_pose: dict[str, float],
    base: str = "auto",
):
    prefab = robotics.load_bot_prefab_from_urdf_file(str(urdf))
    meshes = parse_urdf_meshes(urdf)
    facts = urdf_facts(urdf, prefab, meshes)
    fixed_base = facts["fixed_base"] if base == "auto" else base == "fixed"
    if facts["mimics"]:
        print(f"mimic joints are not imported, their dofs are independent in the package: {facts['mimics']}")
    if facts["rotated_inertials"]:
        print(
            "the importer keeps the inertia tensor in the inertial frame and drops the rotation of "
            f"<inertial><origin rpy>, so these links' tensors are misoriented: {facts['rotated_inertials']}"
        )
    fold_shapeless_links(prefab, meshes)
    out_dir.mkdir(parents=True, exist_ok=True)
    links = []
    for i in range(len(prefab.links)):
        link = prefab.links[i]
        entry = {"name": link.name}
        if link.parent_link >= 0:
            entry["parentLink"] = int(link.parent_link)
        if link.mass is not None:  # frames without <inertial> stay massless
            entry["mass"] = float(link.mass)
            entry["centerOfMass"] = vec(link.center_of_mass)
            entry["momentOfInertia"] = vec(link.moment_of_inertia)
        refs = meshes.get(link.name, {"visual": None, "collision": None, "color": None})
        if refs["collision"] is not None:
            source = resolve_mesh(refs["collision"], urdf.parent, mesh_dir)
            target = out_dir / "collision" / f"{link.name}_collision.stl"
            note = write_collision(source, target, link.shape_scale)
            print(f"  {link.name}: {note}")
            entry["shape"] = f"collision/{target.name}"
            if not is_identity(link.shape_rotation, link.shape_translation):
                entry["shapeRotation"] = vec(link.shape_rotation)
                entry["shapeTranslation"] = vec(link.shape_translation)
        if refs["visual"] is not None:
            source = resolve_mesh(refs["visual"], urdf.parent, mesh_dir)
            target = out_dir / "render" / f"{link.name}_render.glb"
            bounds = write_glb(source, target, refs["color"], link.render_model_scale)
            entry["renderModel"] = f"render/{target.name}"
            if not is_identity(link.render_model_rotation, link.render_model_translation):
                entry["renderModelRotation"] = vec(link.render_model_rotation)
                entry["renderModelTranslation"] = vec(link.render_model_translation)
            if i == 0:
                print(f"root link {link.name}: visual mesh bounds min {bounds[0].round(4)} max {bounds[1].round(4)}")
        links.append(entry)
    joints = []
    revolute = []
    for i in range(len(prefab.joints)):
        joint = prefab.joints[i]
        entry = {"name": joint.name, "type": joint_type_name(joint.type)}
        if i == 0 and fixed_base and joint.type == physics.ArticulatedJointType.FREE:
            entry["type"] = "Hard"  # fixed base
        transform = joint.parent_link_from_joint
        frame = {}
        if not np.allclose(np.asarray(transform.rotation), [0, 0, 0, 1], atol=1e-12):
            frame["rotation"] = vec(transform.rotation)
        if not np.allclose(np.asarray(transform.translation), 0, atol=1e-12):
            frame["translation"] = vec(transform.translation)
        if frame:
            entry["parentLinkFromJoint"] = frame
        if joint.type in (physics.ArticulatedJointType.REVOLUTE, physics.ArticulatedJointType.PRISMATIC):
            entry["axis"] = vec(joint.axis)
            entry["minLimit"] = vec(joint.min_limit)
            entry["maxLimit"] = vec(joint.max_limit)
            if joint.effort_limit > 0:
                entry["effortLimit"] = float(joint.effort_limit)
            friction = joint.friction
            if friction.viscous > 0 or friction.coulomb > 0:
                entry["friction"] = {"coulomb": float(friction.coulomb), "viscous": float(friction.viscous)}
            if friction.stiction_extra > 0 or friction.stribeck_vel > 0:
                print(f"  {joint.name}: Stribeck friction terms are not written to the package")
            if joint.inertia is not None:
                entry["inertia"] = float(joint.inertia)
            entry["limitStiffness"] = float(joint.limit_stiffness)
            entry["limitDamping"] = float(joint.limit_damping)
            revolute.append(joint.name)
        joints.append(entry)
    unknown = set(default_pose) - set(revolute)
    if unknown:
        raise ValueError(f"default pose names unknown joints: {sorted(unknown)}")
    pose = [default_pose.get(n, 0.0) for n in revolute]
    root = links[0]["name"]
    overrides = [
        {"enable": False, "linkA": root, "linkB": link["name"]}
        for link in links[1:]
        if link.get("parentLink") != 0
    ]
    bot = {"contactOverrides": overrides, "defaultPose": pose, "joints": joints, "links": links, "name": name}
    bot_path = out_dir / f"{name}.superdex_bot"
    bot_path.write_text(json.dumps(bot, indent=2, sort_keys=True) + "\n")
    print(f"base: {'fixed (Hard world joint)' if fixed_base else 'floating (Free world joint)'}")
    print(f"wrote {bot_path}: {len(links)} links, {len(joints)} joints ({len(revolute)} revolute)")
    return prefab, bot_path


def link_transforms(scene, bot) -> dict[str, np.ndarray]:
    actors = []
    scene.for_each_actor(lambda a: actors.append(a) if a.is_nested_link_actor() else None)
    out = {}
    for actor in actors:
        transform = actor.get_root_transform()
        out[actor.get_name().split("/")[-1]] = np.concatenate(
            [np.asarray(transform.translation, dtype=np.float64), np.asarray(transform.rotation.tolist(), dtype=np.float64)]
        )
    return out


def perturbed_pose(prefab, pose: np.ndarray) -> np.ndarray:
    """The pose with every revolute or prismatic dof moved to an interior point of its
    limits (or by a small offset when the joint has no limits)."""
    q = np.array(pose, dtype=np.float64)
    offset = 6 if prefab.joints[0].type == physics.ArticulatedJointType.FREE else 0
    k = 0
    single_dof = (physics.ArticulatedJointType.REVOLUTE, physics.ArticulatedJointType.PRISMATIC)
    movable = [prefab.joints[i] for i in range(len(prefab.joints)) if prefab.joints[i].type in single_dof]
    for joint in movable:
        axis = np.asarray(joint.axis, dtype=np.float64)
        lo = float(np.asarray(joint.min_limit, dtype=np.float64) @ axis)
        hi = float(np.asarray(joint.max_limit, dtype=np.float64) @ axis)
        fraction = 0.3 + 0.4 * k / max(len(movable) - 1, 1)
        q[offset + k] = lo + fraction * (hi - lo) if hi > lo else 0.1 + 0.05 * k
        k += 1
    return q


def verify(urdf_prefab, bot_path: pathlib.Path) -> None:
    reloaded = robotics.load_bot_prefab_from_file(str(bot_path))
    if reloaded.joints[0].type != urdf_prefab.joints[0].type:
        # The package may fix the base the URDF import left floating; compare the
        # kinematics with the same world joint on both sides.
        joint = urdf_prefab.joints[0]
        joint.type = reloaded.joints[0].type
        urdf_prefab.joints[0] = joint
    for i in range(len(reloaded.links)):
        link = reloaded.links[i]
        link.collider_type = physics.ColliderType.NONE
        reloaded.links[i] = link
    for i in range(len(urdf_prefab.links)):
        link = urdf_prefab.links[i]
        link.collider_type = physics.ColliderType.NONE
        # The comparison is about kinematics; the package's collision meshes stand in for
        # upstream meshes the engine may reject (a zero-area triangle, an open surface).
        link.shape_file = reloaded.links[i].shape_file
        urdf_prefab.links[i] = link
    poses = {}
    for label, prefab in (("urdf", urdf_prefab), ("package", reloaded)):
        scene = physics.create_scene(label)
        scene.set_gravity([0.0, 0.0, 0.0])
        context = robotics.create_context()
        bot = robotics.create_bot(scene, prefab, context)
        actor = bot.get_articulated_actor()
        q = np.zeros(actor.get_num_dofs())
        actor.get_articulated_pose(q)
        # Move every joint inside its limits so that axes and offsets are compared, not only
        # the rest pose; the root's dofs (a Free world joint) stay where they are.
        q[:] = perturbed_pose(prefab, q)
        actor.set_articulated_pose_from_joints(q)
        scene.step(1e-3)
        poses[label] = (actor.get_num_dofs(), link_transforms(scene, bot))
        robotics.destroy_bot(scene, bot)
        physics.destroy_scene(scene)
        del context
    (n_urdf, t_urdf), (n_pkg, t_pkg) = poses["urdf"], poses["package"]
    if n_urdf != n_pkg or set(t_urdf) != set(t_pkg):
        raise RuntimeError(f"package differs from the URDF: {n_urdf} vs {n_pkg} dofs, links {sorted(set(t_urdf) ^ set(t_pkg))}")
    worst = max(float(np.abs(t_urdf[k] - t_pkg[k]).max()) for k in t_urdf)
    if worst > 1e-9:
        raise RuntimeError(f"package differs from the URDF: link transforms differ by up to {worst:.2e}")
    print(f"verified: {n_pkg} dofs, {len(t_pkg)} link transforms match the URDF import to {worst:.1e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("urdf", type=pathlib.Path)
    parser.add_argument("out_dir", type=pathlib.Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--mesh-dir", type=pathlib.Path, default=None)
    parser.add_argument("--default-pose", nargs="*", default=[], help="JOINT=VALUE pairs [rad]")
    parser.add_argument(
        "--base",
        choices=["auto", "fixed", "floating"],
        default="auto",
        help="fixed base (Hard world joint) or floating (Free); auto: fixed for a bare world root (default)",
    )
    args = parser.parse_args()
    default_pose = {}
    for item in args.default_pose:
        key, value = item.split("=")
        default_pose[key] = float(value)
    physics.initialize(num_worker_threads=0)
    try:
        prefab, bot_path = convert(
            args.urdf.resolve(), args.out_dir.resolve(), args.name, args.mesh_dir, default_pose, args.base
        )
        verify(prefab, bot_path)
    finally:
        physics.shutdown()


if __name__ == "__main__":
    main()
