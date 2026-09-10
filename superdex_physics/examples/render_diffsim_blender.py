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
"""Render the frames a differentiable-simulation demo exported with Blender.

The demos (``example_diffsim_video.py``, ``example_diffsim_robot_video.py``) record their
videos with the built-in viewer. With ``--export-scenes`` they also write, next to each
video, ``<video>.scene.json`` and ``<video>.scene.npz``: every body's mesh (the robot's
render models with their textures, or the physics surface mesh) with a world transform per
frame, the soft bodies' vertices per frame, the rod centerlines, the tracked point, the
target and the camera. This script rebuilds that in Blender, lights it, renders every
frame with Cycles and assembles the same video with the same captions.

Three steps, the last one combining the first two::

    blender -b --python render_diffsim_blender.py -- render VIDEO.scene --frames-dir DIR
    python render_diffsim_blender.py compose VIDEO.scene --frames-dir DIR --output OUT.mp4
    python render_diffsim_blender.py video VIDEO.scene --output OUT.mp4

``render`` runs inside Blender (4.1 or newer) and writes one PNG per recorded frame;
``compose`` runs in the project's Python environment (OpenCV, imageio) and adds the title
and captions, repeats the frames each capture holds and encodes the MP4; ``video`` runs
Blender as a subprocess and then composes. Rendering uses the GPU when Cycles finds one.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess
import sys

import numpy as np

try:
    import bpy  # noqa: F401

    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False


def load_scene(stem: pathlib.Path):
    stem = pathlib.Path(stem)
    if stem.suffix == ".scene":
        stem = stem.with_suffix("")
    with open(stem.with_suffix(".scene.json")) as handle:
        meta = json.load(handle)
    arrays = dict(np.load(stem.with_suffix(".scene.npz")))
    return stem, meta, arrays


# ---------------------------------------------------------------------------
# Inside Blender
# ---------------------------------------------------------------------------


def _material(name: str, base_color, roughness: float, metallic: float = 0.0, emission=None,
              texture_path: pathlib.Path | None = None, subsurface: float = 0.0, specular: float | None = None):
    import bpy

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*base_color[:3], 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if specular is not None and "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = specular
    if subsurface > 0.0 and "Subsurface Weight" in bsdf.inputs:
        bsdf.inputs["Subsurface Weight"].default_value = subsurface
        if "Subsurface Radius" in bsdf.inputs:
            bsdf.inputs["Subsurface Radius"].default_value = tuple(0.02 * c for c in base_color[:3])
    if emission is not None:
        bsdf.inputs["Emission Color"].default_value = (*emission[:3], 1.0)
        bsdf.inputs["Emission Strength"].default_value = 1.0
    if texture_path is not None and texture_path.is_file():
        image = bpy.data.images.load(str(texture_path))
        node = nodes.new("ShaderNodeTexImage")
        node.image = image
        links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
    return material


def _body_material(meta: dict, stem: pathlib.Path):
    name = meta["name"].lower()
    material = meta.get("material") or {}
    if meta["kind"] == "soft":
        base = material.get("base_color") or (0.05, 0.42, 0.12)
        return _material(meta["name"], base, 0.35, specular=0.1)
    if meta["source"] == "render_model":
        base = material.get("base_color") or [0.82, 0.82, 0.8]
        roughness = material.get("roughness")
        metallic = material.get("metallic") or 0.0
        dark = any(key in name for key in ("finger", "knuckle", "pad", "coupler", "2f_85"))
        if roughness is None:
            roughness = 0.6 if dark else 0.4
        texture = stem.parent / f"{stem.name}.scene.body{meta['index']}.png" if meta.get("texture") else None
        return _material(meta["name"], base, roughness, metallic, texture_path=texture)
    # Physics meshes: the manipulated objects. An exported base color wins; otherwise the
    # manipulated cube/box is the accent red and anything else a neutral grey.
    base = material.get("base_color")
    if base is not None:
        return _material(meta["name"], base, 0.4)
    if any(key in name for key in ("cube", "box", "block", "target")):
        return _material(meta["name"], (0.72, 0.2, 0.02), 0.35)
    return _material(meta["name"], (0.6, 0.62, 0.66), 0.5)


def _mesh_object(name: str, vertices: np.ndarray, faces: np.ndarray, material, uv=None, smooth=True):
    import bpy

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(map(float, v)) for v in vertices], [], [tuple(map(int, f)) for f in faces])
    mesh.update()
    if uv is not None:
        layer = mesh.uv_layers.new(name="UVMap")
        flat = np.asarray(uv, dtype=np.float32)
        loop_vertices = np.zeros(len(mesh.loops), dtype=np.int64)
        mesh.loops.foreach_get("vertex_index", loop_vertices)
        layer.data.foreach_set("uv", flat[loop_vertices].ravel())
    if smooth:
        mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    mesh.materials.append(material)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    if smooth:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        for operator in ("shade_smooth_by_angle", "shade_auto_smooth"):
            if hasattr(bpy.ops.object, operator):
                try:
                    getattr(bpy.ops.object, operator)(angle=math.radians(35.0))
                except Exception:
                    pass
                break
        obj.select_set(False)
    return obj


def _curve_object(name: str, radius: float, material):
    import bpy

    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = radius
    curve.bevel_resolution = 6
    curve.use_fill_caps = True
    curve.materials.append(material)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _set_polyline(obj, points: np.ndarray) -> None:
    curve = obj.data
    for spline in list(curve.splines):
        curve.splines.remove(spline)
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        obj.hide_render = True
        return
    obj.hide_render = False
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    flat = np.concatenate([points, np.ones((len(points), 1))], axis=1).ravel()
    spline.points.foreach_set("co", flat)


def _lights_and_camera(meta: dict, size) -> None:
    import bpy
    from mathutils import Vector

    scene = bpy.context.scene
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    background = nodes["Background"]
    if meta.get("ground"):
        # Daylight over the ground plane.
        sky = nodes.new("ShaderNodeTexSky")
        sky.sky_type = "NISHITA"
        sky.sun_disc = False
        sky.sun_elevation = math.radians(55.0)
        sky.sun_rotation = math.radians(200.0)
        sky.altitude = 200.0
        background.inputs["Strength"].default_value = 0.45
        links.new(sky.outputs["Color"], background.inputs["Color"])
    else:
        # No ground in the scene (the tendon finger floats): a plain studio backdrop.
        background.inputs["Color"].default_value = (0.82, 0.83, 0.85, 1.0)
        background.inputs["Strength"].default_value = 0.9

    sun = bpy.data.lights.new("Sun", type="SUN")
    sun.energy = 4.5
    sun.angle = math.radians(2.5)
    sun_obj = bpy.data.objects.new("Sun", sun)
    scene.collection.objects.link(sun_obj)
    sun_obj.rotation_euler = (math.radians(35.0), math.radians(10.0), math.radians(200.0))

    fill = bpy.data.lights.new("Fill", type="AREA")
    fill.energy = 250.0
    fill.size = 3.0
    fill_obj = bpy.data.objects.new("Fill", fill)
    scene.collection.objects.link(fill_obj)
    look_from = Vector(meta["look_from"])
    look_at = Vector(meta["look_at"])
    fill_obj.location = look_from + Vector((0.0, 0.0, 2.0))
    fill_obj.rotation_euler = (look_at - fill_obj.location).to_track_quat("-Z", "Y").to_euler()

    camera = bpy.data.cameras.new("Camera")
    camera.sensor_fit = "VERTICAL"
    # A little tighter than the viewer's 45 degree vertical field of view; the framing of
    # the demos leaves room for it.
    camera.angle_y = math.radians(36.0)
    camera_obj = bpy.data.objects.new("Camera", camera)
    scene.collection.objects.link(camera_obj)
    camera_obj.location = look_from
    camera_obj.rotation_euler = (look_at - look_from).to_track_quat("-Z", "Y").to_euler()
    scene.camera = camera_obj
    scene.render.resolution_x, scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    # The default is AgX; a demo whose colours AgX would wash out (the FEM jellies) exports
    # its own view transform and exposure in the scene meta.
    if meta.get("view_transform"):
        scene.view_settings.view_transform = meta["view_transform"]
        if meta.get("view_exposure") is not None:
            scene.view_settings.exposure = float(meta["view_exposure"])
    else:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Medium High Contrast"


def _ground() -> None:
    import bpy

    material = _material("Ground", (0.52, 0.52, 0.5), 0.75)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 40.0
    noise.inputs["Detail"].default_value = 6.0
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.08
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], nodes["Principled BSDF"].inputs["Normal"])
    half = 80.0
    vertices = np.array([[-half, -half, 0.0], [half, -half, 0.0], [half, half, 0.0], [-half, half, 0.0]])
    _mesh_object("Ground", vertices, np.array([[0, 1, 2], [0, 2, 3]]), material, smooth=False)


def _render_settings(samples: int) -> None:
    import bpy

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.use_denoising = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    preferences = bpy.context.preferences.addons.get("cycles")
    if preferences is not None:
        cycles = preferences.preferences
        for device_type in ("OPTIX", "CUDA", "HIP", "METAL"):
            try:
                cycles.compute_device_type = device_type
                cycles.get_devices()
            except Exception:
                continue
            devices = [d for d in cycles.devices if d.type != "CPU"]
            if devices:
                for device in cycles.devices:
                    device.use = device.type != "CPU"
                scene.cycles.device = "GPU"
                break


def render(stem: pathlib.Path, frames_dir: pathlib.Path, samples: int, size, only=None) -> None:
    import bpy
    from mathutils import Matrix

    stem, meta, arrays = load_scene(stem)
    frames_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    _render_settings(samples)
    _lights_and_camera(meta, size)
    if meta.get("ground"):
        _ground()

    objects = {}
    for body in meta["bodies"]:
        index = body["index"]
        material = _body_material(body, stem)
        faces = arrays[f"body{index}_faces"]
        if body["kind"] == "soft":
            vertices = arrays[f"body{index}_vertex_frames"][0]
            objects[index] = _mesh_object(body["name"], vertices, faces, material, smooth=True)
        else:
            vertices = arrays[f"body{index}_vertices"]
            uv = arrays.get(f"body{index}_uv")
            smooth = body["source"] == "render_model"
            objects[index] = _mesh_object(body["name"], vertices, faces, material, uv=uv, smooth=smooth)

    if meta.get("target") is not None:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.035, location=meta["target"], segments=48, ring_count=24)
        target = bpy.context.active_object
        target.name = "Target"
        target.data.materials.append(_material("Target", (0.85, 0.12, 0.1), 0.4, emission=(0.4, 0.05, 0.04)))
        target.data.polygons.foreach_set("use_smooth", [True] * len(target.data.polygons))

    trail = _curve_object("Trail", 0.007, _material("Trail", (0.08, 0.32, 0.95), 0.4, emission=(0.04, 0.16, 0.6)))
    curve_objects = {}

    scene = bpy.context.scene
    trail_points = []
    for frame_index, frame in enumerate(meta["frames"]):
        if frame["new_iteration"]:
            trail_points = []
        if frame["tracked"] is not None:
            trail_points.append(frame["tracked"])
        if only is not None and frame_index not in only:
            continue
        _set_polyline(trail, np.asarray(trail_points))
        for body in meta["bodies"]:
            index = body["index"]
            obj = objects[index]
            if body["kind"] == "soft":
                vertices = arrays[f"body{index}_vertex_frames"][frame_index]
                obj.data.vertices.foreach_set("co", np.asarray(vertices, dtype=np.float32).ravel())
                obj.data.update()
            else:
                transforms = arrays[f"body{index}_transforms"]
                transform = transforms[0] if body["static"] else transforms[frame_index]
                obj.matrix_world = Matrix([list(map(float, row)) for row in transform])
        for curve in frame["curves"]:
            name = curve["name"]
            if name not in curve_objects:
                curve_objects[name] = _curve_object(name, curve["radius"], _material(name, curve["color"], 0.5))
            _set_polyline(curve_objects[name], np.asarray(curve["points"]))
        scene.render.filepath = str(frames_dir / f"frame_{frame_index:05d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"rendered frame {frame_index + 1}/{len(meta['frames'])}", flush=True)


# ---------------------------------------------------------------------------
# In the project's Python environment
# ---------------------------------------------------------------------------


def compose(stem: pathlib.Path, frames_dir: pathlib.Path, output: pathlib.Path) -> None:
    import cv2
    import imageio.v3 as iio

    stem, meta, _ = load_scene(stem)
    frames = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for index, frame in enumerate(meta["frames"]):
        image = np.ascontiguousarray(iio.imread(frames_dir / f"frame_{index:05d}.png")[..., :3])
        scale = image.shape[0] / 540.0
        cv2.putText(image, meta["title"], (int(18 * scale), int(34 * scale)), font, 0.8 * scale,
                    (20, 20, 20), max(1, int(2 * scale)), cv2.LINE_AA)
        y = 66 * scale
        for line in frame["caption"].split("\n"):
            cv2.putText(image, line, (int(18 * scale), int(y)), font, 0.6 * scale, (40, 40, 40),
                        max(1, int(scale)), cv2.LINE_AA)
            y += 26 * scale
        for _ in range(frame["hold"]):
            frames.append(image)
    output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output, np.stack(frames), fps=meta["fps"], codec="libx264", macro_block_size=1)
    print(f"wrote {output} ({len(frames)} frames, {len(frames) / meta['fps']:.1f} s)")


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("mode", choices=["render", "compose", "video"])
    parser.add_argument("scene", type=pathlib.Path, help="<video>.scene (the .json/.npz pair)")
    parser.add_argument("--frames-dir", type=pathlib.Path, default=None)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--size", type=int, nargs=2, default=(1280, 720))
    parser.add_argument("--blender", default=os.environ.get("BLENDER", "blender"))
    parser.add_argument("--only", type=int, nargs="*", default=None, help="render these frames only (a preview)")
    args = parser.parse_args(argv)
    stem = args.scene if args.scene.suffix != ".scene" else args.scene.with_suffix("")
    frames_dir = args.frames_dir or stem.parent / f"{stem.name}_frames"
    output = args.output or stem.parent / f"{stem.name}_blender.mp4"
    if args.mode == "render":
        if not IN_BLENDER:
            raise SystemExit("run the render step inside Blender: blender -b --python ... -- render ...")
        render(stem, frames_dir, args.samples, tuple(args.size), args.only)
    elif args.mode == "compose":
        compose(stem, frames_dir, output)
    else:
        command = [args.blender, "-b", "--python", os.path.abspath(__file__), "--", "render", str(stem),
                   "--frames-dir", str(frames_dir), "--samples", str(args.samples),
                   "--size", str(args.size[0]), str(args.size[1])]
        subprocess.run(command, check=True)
        compose(stem, frames_dir, output)


if __name__ == "__main__":
    main()
