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

"""Example: optimization through the differentiable simulator, recorded as video.

Two tasks are solved by gradient-based optimization where every gradient comes
from the engine's discrete adjoint (``superdex.physics.diffsim``), and the
whole optimization is rendered offscreen with the built-in polyscope viewer and
written to MP4 (one file per task):

1. ``rigid_throw.mp4`` - a rigid cube is thrown from a fixed spot. Gradient
   descent on its initial velocity makes it come to rest on a target after
   0.6 s, through impact and frictional sliding on the ground.
2. ``soft_landing.mp4`` - a soft (FEM, Neo-Hookean) jelly cube is launched
   towards a target. Gradient descent (``torch.optim.SGD`` on the launch
   velocity, gradients from the ``superdex.physics.diffsim_torch`` autograd
   bridge) makes its centroid come to rest on the target; the gradient flows
   through the elastic dynamics and the soft-body contact adjoint. (Plain
   descent converges monotonically here, 0.34 -> 2e-8 in 40 iterations;
   Adam's per-coordinate normalization overshoots the narrow valley and
   oscillates around 1e-3.)

Each video shows a selection of optimization iterations back to back: the
rollout of that iteration, the trail of the tracked point, the target, and the
current loss. The loss curves are also written as PNG.

On the soft task the engine may log ``Zero Preconditioner-dot product`` from
its PCG adjoint solve on some steps: the PSD-projected approximate Hessian
used as preconditioner can be singular on a contact island, PCG aborts and the
engine falls back to MINRES. The per-iteration ``adjoint residual`` printed by
this script (about 1e-10) confirms every adjoint solve still converged.

Requirements: double precision (selected below, before the first physics
import), ``polyscope`` (offscreen rendering), ``imageio`` with ffmpeg, OpenCV
(text overlays) and PyTorch (task 2). Run from anywhere::

    python example_diffsim_video.py --output-dir ./diffsim_videos
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

# Precision is process-wide and resolved at the first physics import.
os.environ.setdefault("SUPERDEX_PRECISION", "double")

import cv2
import imageio.v3 as iio
import numpy as np
import superdex.physics as physics
from superdex.physics.utils import render_model_registry
from superdex.physics.utils.transformations import make_transform, transformrt_to_numpy
from superdex.physics.viewer import Viewer, ViewerCfg

diffsim = physics.diffsim

FRAME_SIZE = (960, 540)
FPS = 25
# With --export-scenes, every recorded video also gets a <video>.scene.json/.npz pair that
# render_diffsim_blender.py turns into a Blender rendering of the same frames.
EXPORT_SCENES = False
GRAVITY = [0.0, 0.0, -9.81]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def box_tet_mesh(size: float, cells: int):
    """A structured tetrahedral mesh of a cube: ``cells^3`` hexahedra, each split
    into six tetrahedra (Kuhn triangulation). Returns (coordinates, connectivity)
    flattened the way ``create_tet_mesh_shape`` expects them."""
    n = cells + 1
    axis = np.linspace(-0.5 * size, 0.5 * size, n)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)  # (n,n,n,3)
    coordinates = grid.reshape(-1, 3)

    def node(i, j, k):
        return (i * n + j) * n + k

    kuhn = [
        (0, 1, 3, 7),
        (0, 1, 5, 7),
        (0, 4, 5, 7),
        (0, 4, 6, 7),
        (0, 2, 6, 7),
        (0, 2, 3, 7),
    ]
    tets = []
    for i in range(cells):
        for j in range(cells):
            for k in range(cells):
                corners = [
                    node(i + di, j + dj, k + dk)
                    for di in (0, 1)
                    for dj in (0, 1)
                    for dk in (0, 1)
                ]  # index bits: di*4 + dj*2 + dk
                for tet in kuhn:
                    tets.append([corners[c] for c in tet])
    connectivity = np.asarray(tets, dtype=np.int32)
    # Enforce positive orientation.
    p = coordinates[connectivity]
    volume = np.einsum(
        "ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]
    )
    flip = volume < 0.0
    connectivity[flip, 2], connectivity[flip, 3] = (
        connectivity[flip, 3].copy(),
        connectivity[flip, 2].copy(),
    )
    return coordinates.reshape(-1).astype(np.float64), connectivity.reshape(-1)


# Minimal rigid cube tet mesh (side length 0.2), as in the other diffsim examples.
# fmt: off
CUBE_COORDS = np.array([
    -0.1, -0.1, -0.1,  +0.1, -0.1, -0.1,  -0.1, +0.1, -0.1,  +0.1, +0.1, -0.1,
    -0.1, -0.1, +0.1,  +0.1, -0.1, +0.1,  -0.1, +0.1, +0.1,  +0.1, +0.1, +0.1,
], dtype=np.float64)
CUBE_CONN = np.array([
    0, 1, 2, 4,  6, 7, 4, 2,  5, 4, 7, 1,  3, 2, 1, 7,  1, 2, 4, 7,
], dtype=np.int32)
# fmt: on


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def _load_render_model(glb_path: str, scale) -> dict:
    """The geometry and material of a registered render model, in the frame the viewer
    draws it in (per-axis scale in the model frame, then the model's Y/Z axes unflipped
    into the shape frame, as GlbActorRenderer does)."""
    import trimesh

    loaded = trimesh.load(glb_path, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"render model is not a single mesh: {glb_path}")
    unflip = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    vertices = np.asarray(loaded.vertices, dtype=np.float64) * np.asarray(scale, dtype=np.float64)
    vertices = vertices @ unflip.T
    faces = np.asarray(loaded.faces, dtype=np.int32).reshape(-1, 3)
    material = {"base_color": None, "metallic": None, "roughness": None}
    uv = None
    texture = None
    visual = loaded.visual
    if hasattr(visual, "uv") and visual.uv is not None and len(visual.uv) == len(vertices):
        uv = np.asarray(visual.uv, dtype=np.float32)
    mat = getattr(visual, "material", None)
    if mat is not None:
        color = getattr(mat, "baseColorFactor", None)
        if color is None:
            color = getattr(mat, "diffuse", None)
        if color is not None:
            color = np.asarray(color, dtype=np.float64).reshape(-1)[:4]
            if color.max() > 1.0:
                color = color / 255.0
            material["base_color"] = [float(c) for c in color]
        for key in ("metallicFactor", "roughnessFactor"):
            value = getattr(mat, key, None)
            if value is not None:
                material[key.replace("Factor", "")] = float(value)
        texture = getattr(mat, "baseColorTexture", None)
        if texture is None:
            texture = getattr(mat, "image", None)
    return {"vertices": vertices, "faces": faces, "uv": uv, "texture": texture, "material": material}


class SceneExport:
    """Everything a re-rendering of the recorded frames needs, gathered while the
    polyscope video is recorded: each body's mesh (the registered render model with its
    texture, or the physics surface mesh) and its world transform per frame, the soft
    bodies' vertices per frame, the curves, the trail, the target and the camera.
    ``Recorder.write`` stores it as ``<video>.scene.json`` plus ``<video>.scene.npz`` (and
    the textures as PNG files); ``render_diffsim_blender.py`` renders it with Blender."""

    def __init__(self, scene, target, look_from, look_at, title: str):
        self.scene = scene
        self.meta = {
            "title": title,
            "target": [float(v) for v in target],
            "look_from": [float(v) for v in look_from],
            "look_at": [float(v) for v in look_at],
            "fps": FPS,
            "size": list(FRAME_SIZE),
            "ground": False,
            "bodies": [],
            "frames": [],
        }
        self.arrays: dict[str, np.ndarray] = {}
        self.textures: dict[int, object] = {}
        self.bodies: list[dict] = []
        self.new_iteration = True
        scene.for_each_actor(self._add_actor)

    def _add_actor(self, actor) -> None:
        index = len(self.bodies)
        meta = {"name": actor.get_name(), "index": index, "static": bool(actor.is_static())}
        entry = render_model_registry.get(self.scene.get_handle(), actor.get_handle())
        if entry is not None:
            model = _load_render_model(entry.glb_path, entry.scale)
            meta.update(kind="rigid", source="render_model", material=model["material"])
            self.arrays[f"body{index}_vertices"] = model["vertices"].astype(np.float32)
            self.arrays[f"body{index}_faces"] = model["faces"]
            if model["uv"] is not None:
                self.arrays[f"body{index}_uv"] = model["uv"]
            if model["texture"] is not None:
                self.textures[index] = model["texture"]
                meta["texture"] = True
            self.bodies.append({"actor": actor, "local": entry.local_transform, "meta": meta, "transforms": []})
            self.meta["bodies"].append(meta)
            return
        surface = actor.get_surface_mesh()
        if surface.is_empty():
            # The demos' ground is an infinite plane at z = 0; rods are drawn from their
            # centerlines, which the curves carry.
            if actor.is_static():
                self.meta["ground"] = True
            return
        faces = np.asarray(surface.connectivity, dtype=np.int32).reshape(-1, 3)
        soft = actor.get_type() in (physics.ActorType.SOFT, physics.ActorType.SHELL)
        meta.update(
            kind="soft" if soft else "rigid",
            source="physics_mesh",
            material={"base_color": None, "metallic": None, "roughness": None},
        )
        self.arrays[f"body{index}_faces"] = faces
        body = {"actor": actor, "local": None, "meta": meta}
        if soft:
            body["vertex_frames"] = []
        else:
            self.arrays[f"body{index}_vertices"] = self._local_vertices(actor).astype(np.float32)
            body["transforms"] = []
        self.bodies.append(body)
        self.meta["bodies"].append(meta)

    @staticmethod
    def _local_vertices(actor) -> np.ndarray:
        actor.register_query_and_compute(physics.QueryType.SURFACE_NODE_POSITIONS)
        return np.asarray(actor.get_surface_mesh_node_positions_local(), dtype=np.float64).reshape(-1, 3)

    @staticmethod
    def _world_from(actor, local) -> np.ndarray:
        transform = actor.get_root_transform()
        if local is not None:
            transform = transform * local
        position, rotvec = transformrt_to_numpy(transform)
        return np.asarray(make_transform(position, rotvec), dtype=np.float64)

    def begin_iteration(self) -> None:
        self.new_iteration = True

    def capture(self, tracked_point, caption: str, hold: int, curves) -> None:
        frame = {
            "caption": caption,
            "hold": int(hold),
            "tracked": [float(v) for v in np.asarray(tracked_point, dtype=np.float64)],
            "new_iteration": self.new_iteration,
            "curves": [],
        }
        self.new_iteration = False
        for body in self.bodies:
            actor = body["actor"]
            if body["meta"]["kind"] == "soft":
                local = self._local_vertices(actor)
                if actor.has_root_transform():
                    world_from_local = self._world_from(actor, None)
                    local = local @ world_from_local[:3, :3].T + world_from_local[:3, 3]
                body["vertex_frames"].append(local.astype(np.float32))
            elif not body["meta"]["static"] or not body["transforms"]:
                body["transforms"].append(self._world_from(actor, body["local"]))
        if curves is not None:
            for name, points, edges, radius, color in curves():
                frame["curves"].append(
                    {
                        "name": str(name),
                        "points": np.asarray(points, dtype=np.float64).tolist(),
                        "edges": np.asarray(edges, dtype=np.int64).tolist(),
                        "radius": float(radius),
                        "color": [float(c) for c in color],
                    }
                )
        self.meta["frames"].append(frame)

    def write(self, stem: pathlib.Path) -> None:
        arrays = dict(self.arrays)
        for body in self.bodies:
            index = body["meta"]["index"]
            if body["meta"]["kind"] == "soft":
                arrays[f"body{index}_vertex_frames"] = np.stack(body["vertex_frames"])
            else:
                arrays[f"body{index}_transforms"] = np.stack(body["transforms"])
        np.savez_compressed(stem.with_suffix(".scene.npz"), **arrays)
        for index, texture in self.textures.items():
            texture.save(stem.parent / f"{stem.name}.scene.body{index}.png")
        with open(stem.with_suffix(".scene.json"), "w") as handle:
            json.dump(self.meta, handle)
        print(f"wrote {stem.with_suffix('.scene.json')} ({len(self.meta['frames'])} frames)")


class Recorder:
    """Offscreen viewer + MP4 writer with text overlays and a tracked-point trail.
    With ``EXPORT_SCENES`` set, the recorded frames are also exported for Blender
    (see :class:`SceneExport`)."""

    def __init__(self, scene, target, look_from, look_at, title: str, curves=None):
        # The scene is Z-up FLU (X forward, Y left, Z up) - the "ros" preset.
        self.viewer = Viewer(
            ViewerCfg(offscreen=True, size=FRAME_SIZE, coordinate_system="ros")
        )
        self.viewer.set_scene(scene)
        self.viewer.add_point_cloud(
            "target", np.asarray([target]), radius=0.035, color=[0.9, 0.15, 0.15]
        )
        self.viewer.set_camera_view(look_from=look_from, look_at=look_at)
        self.title = title
        # Optional extra geometry the viewer does not draw itself (rod centerlines):
        # a callable returning (name, points, edges, radius, color) tuples per frame.
        self.curves = curves
        self.frames: list[np.ndarray] = []
        self.trail: list[np.ndarray] = []
        self.export = (
            SceneExport(scene, target, look_from, look_at, title) if EXPORT_SCENES else None
        )

    def begin_iteration(self) -> None:
        self.trail = []
        if self.export is not None:
            self.export.begin_iteration()

    def capture(self, tracked_point, caption: str, hold: int = 1) -> None:
        self.trail.append(np.asarray(tracked_point, dtype=np.float64))
        if len(self.trail) >= 2:
            pts = np.asarray(self.trail)
            edges = np.stack([np.arange(len(pts) - 1), np.arange(1, len(pts))], axis=1)
            self.viewer.add_curve_network(
                "trail", pts, edges, radius=0.006, color=[0.1, 0.35, 0.9]
            )
        if self.curves is not None:
            for name, pts, edges, radius, color in self.curves():
                self.viewer.add_curve_network(name, pts, edges, radius=radius, color=color)
        frame = np.ascontiguousarray(np.asarray(self.viewer.render())[..., :3])
        self._overlay(frame, caption)
        for _ in range(hold):
            self.frames.append(frame)
        if self.export is not None:
            self.export.capture(tracked_point, caption, hold, self.curves)

    def _overlay(self, frame: np.ndarray, caption: str) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, self.title, (18, 34), font, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
        y = 66
        for line in caption.split("\n"):
            cv2.putText(frame, line, (18, y), font, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
            y += 26

    def write(self, path: pathlib.Path) -> None:
        iio.imwrite(path, np.stack(self.frames), fps=FPS, codec="libx264", macro_block_size=1)
        print(f"wrote {path} ({len(self.frames)} frames, {len(self.frames) / FPS:.1f} s)")
        if self.export is not None:
            self.export.write(pathlib.Path(path).with_suffix(""))
        self.viewer.close()


def save_loss_curve(path: pathlib.Path, losses, title: str) -> None:
    """Loss-vs-iteration plot drawn with OpenCV (no matplotlib dependency)."""
    width, height, margin = 640, 360, 50
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    values = np.log10(np.maximum(np.asarray(losses, dtype=np.float64), 1e-12))
    lo, hi = values.min() - 0.2, values.max() + 0.2
    xs = margin + (width - 2 * margin) * np.arange(len(values)) / max(len(values) - 1, 1)
    ys = height - margin - (height - 2 * margin) * (values - lo) / (hi - lo)
    cv2.rectangle(image, (margin, margin), (width - margin, height - margin), (200, 200, 200), 1)
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        cv2.line(image, (int(x0), int(y0)), (int(x1), int(y1)), (200, 60, 30), 2, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, title, (margin, 32), font, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(image, "iteration", (width // 2 - 40, height - 14), font, 0.5, (60, 60, 60), 1)
    cv2.putText(image, "log10 loss", (6, margin - 10), font, 0.5, (60, 60, 60), 1)
    cv2.putText(image, f"{10 ** lo:.1e}", (2, height - margin), font, 0.4, (60, 60, 60), 1)
    cv2.putText(image, f"{10 ** hi:.1e}", (2, margin + 12), font, 0.4, (60, 60, 60), 1)
    iio.imwrite(path, image)


def tighten_solvers(scene) -> None:
    """A tight forward Newton solve and a tight adjoint solve: the adjoint assumes
    the step equations hold exactly, and friction gradients degrade visibly with
    the default 1e-3 tolerances (see the sysid example)."""
    solver = scene.get_solver_params()
    newton = solver.non_linear_solver
    newton.max_iter = 200
    newton.abs_tol = 1e-12
    newton.rel_tol = 1e-12
    solver.non_linear_solver = newton
    scene.set_solver_params(solver)
    params = diffsim.get_back_propagation_solver_params(scene)
    params.outer_solver_abs_tol = 1e-10
    params.outer_solver_max_iter = 100
    # Keep the inner (preconditioner) solve tighter than the outer one.
    params.inner_solver_abs_tol = 1e-14
    diffsim.set_back_propagation_solver_params(scene, params)


def recorded_iterations(num_iterations: int) -> set[int]:
    picks = {0, 1, 2, 3, 5, 8, 12, 18, 25, 35, 50, 70, 100}
    return {i for i in picks if i < num_iterations} | {num_iterations - 1}


# ---------------------------------------------------------------------------
# Task 1: rigid throw with the per-step adjoint API
# ---------------------------------------------------------------------------


def task_rigid_throw(output_dir: pathlib.Path, num_iterations: int) -> None:
    dt, num_steps = 0.02, 30
    target = np.array([0.9, 0.25, 0.1])
    learning_rate = 2.0

    scene = physics.create_scene("Differentiable throw")
    scene.set_gravity(GRAVITY)
    contact = physics.ContactParams(penalty_coefficient=1e8, coulomb_friction_coefficient=0.3)
    scene.create_rigid_actor(
        name="ground",
        shape=physics.create_plane_shape(normal=[0, 0, 1], distance=0.0),
        is_static=True,
        contact=contact,
    )
    cube = scene.create_rigid_actor(
        name="cube",
        shape=physics.create_tet_mesh_shape(coordinates=CUBE_COORDS, connectivity=CUBE_CONN),
        density=1000.0,
        contact=contact,
        world_from_local=physics.TransformRT([0.0, 0.0, 0.5]),
    )
    diffsim.make_scene_differentiable(scene)
    tighten_solvers(scene)
    recorder = Recorder(
        scene,
        target,
        look_from=[-0.9, -2.2, 1.3],
        look_at=[0.45, 0.1, 0.15],
        title="Rigid throw: gradient descent on the launch velocity (per-step adjoint)",
    )
    state_init = scene.capture_state()
    velocity = np.zeros(3)
    losses = []
    record = recorded_iterations(num_iterations)

    for iteration in range(num_iterations):
        scene.restore_state(state_init, False)
        cube.set_velocity(velocity, [0.0, 0.0, 0.0])
        recording = iteration in record
        if recording:
            recorder.begin_iteration()
        pre, post = [], []
        for step in range(num_steps):
            pre.append(scene.capture_state())
            scene.step(dt)
            post.append(scene.capture_state())
            if recording:
                com = np.asarray(cube.get_center_of_mass_transform().translation)
                caption = (
                    f"iteration {iteration}   t = {(step + 1) * dt:.2f} s\n"
                    f"v0 = [{velocity[0]:+.2f} {velocity[1]:+.2f} {velocity[2]:+.2f}] m/s"
                    + (f"   loss = {losses[-1]:.4f}" if losses else "")
                )
                recorder.capture(com, caption, hold=(1 if step < num_steps - 1 else 12))

        position = np.asarray(cube.get_center_of_mass_transform().translation)
        displacement = position - target
        loss = 0.5 * float(displacement @ displacement)
        losses.append(loss)

        # Reverse sweep: terminal loss gradient, then newest step first.
        diffsim.reset_back_propagation(scene)
        diffsim.prepare_back_propagate(scene, post[-1], pre[-1])
        grad_output = np.zeros(7)
        grad_output[:3] = displacement
        diffsim.get_center_of_mass_transform_backward(cube, grad_output)
        for i in range(num_steps, 0, -1):
            if i != num_steps:
                diffsim.prepare_back_propagate(scene, post[i - 1], pre[i - 1])
            diffsim.back_propagate(scene)
        grad_linear, grad_angular = np.zeros(3), np.zeros(3)
        diffsim.set_velocity_backward(cube, grad_linear, grad_angular)
        for handle in pre + post:
            scene.release_state(handle)

        print(f"[rigid] iter {iteration:3d}  loss {loss:.6f}  final {position.round(3)}  v0 {velocity.round(3)}")
        velocity -= learning_rate * grad_linear

    recorder.write(output_dir / "rigid_throw.mp4")
    save_loss_curve(output_dir / "rigid_throw_loss.png", losses, "Rigid throw: loss vs iteration")
    physics.destroy_scene(scene)


# ---------------------------------------------------------------------------
# Task 2: soft landing through the torch bridge
# ---------------------------------------------------------------------------


def task_soft_landing(output_dir: pathlib.Path, num_iterations: int) -> None:
    import torch
    from superdex.physics.diffsim_torch import TorchRollout

    dt, num_steps = 0.02, 40
    target = np.array([0.8, -0.2, 0.0])
    coordinates, connectivity = box_tet_mesh(size=0.2, cells=3)
    num_nodes = coordinates.size // 3

    scene = physics.create_scene("Differentiable soft landing")
    scene.set_gravity(GRAVITY)
    contact = physics.ContactParams(penalty_coefficient=1e7, coulomb_friction_coefficient=0.4)
    scene.create_rigid_actor(
        name="ground",
        shape=physics.create_plane_shape(normal=[0, 0, 1], distance=0.0),
        is_static=True,
        contact=contact,
    )
    material = physics.SoftMaterialParams(density=1000.0, mass_damping_coefficient=1.0)
    material.neo_hookean = physics.NeoHookeanMaterialParams(youngs_modulus=4.0e4, poisson_ratio=0.45)
    jelly = scene.create_soft_actor(
        name="jelly",
        shape=physics.create_tet_mesh_shape(coordinates=coordinates, connectivity=connectivity),
        material=material,
        contact=contact,
        world_from_local=physics.TransformRT([0.0, 0.0, 0.35]),
    )
    diffsim.make_scene_differentiable(scene)  # also disables recentering (fixed root frame)
    tighten_solvers(scene)
    rest = coordinates.reshape(-1, 3)
    root = np.array([0.0, 0.0, 0.35])

    def centroid_world() -> np.ndarray:
        return root + (rest + np.asarray(jelly.get_displacements()).reshape(-1, 3)).mean(axis=0)

    class CentroidLoss:
        """0.5 |centroid - target|^2 (the target lies on the ground: z is the
        resting height of the jelly's centroid, which the optimizer cannot change)."""

        ref = target + np.array([0.0, 0.0, 0.1])

        def value(self) -> float:
            d = centroid_world() - self.ref
            return 0.5 * float(d @ d)

        def accumulate_output_grad(self) -> None:
            d = centroid_world() - self.ref
            diffsim.get_displacements_backward(jelly, np.tile(d / num_nodes, num_nodes))

    recorder = Recorder(
        scene,
        target,
        look_from=[-0.8, -2.0, 1.1],
        look_at=[0.4, -0.05, 0.15],
        title="Soft landing: gradient descent on the launch velocity (FEM + contact adjoint, torch)",
    )
    bridge = TorchRollout(
        scene,
        dt=dt,
        num_steps=num_steps,
        initial_state_actors=[jelly],
        terminal_losses=[CentroidLoss()],
    )
    velocity = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([velocity], lr=4.0)
    losses = []
    record = recorded_iterations(num_iterations)
    u0 = torch.zeros(3 * num_nodes, dtype=torch.float64)

    for iteration in range(num_iterations):
        optimizer.zero_grad()
        loss = bridge(initial_states=torch.cat([u0, velocity.repeat(num_nodes)]))
        loss.backward()
        losses.append(float(loss))
        v0 = velocity.detach().numpy().copy()
        result = bridge.last_result
        print(
            f"[soft] iter {iteration:3d}  loss {float(loss):.6f}  v0 {v0.round(3)}  "
            f"|grad| {float(velocity.grad.norm()):.3e}  "
            f"adjoint residual {result.max_adjoint_residual:.1e}"
        )

        if iteration in record:
            # Replay this iteration's rollout for the camera (the bridge restored
            # and re-ran the scene; here we run it once more, without gradients).
            scene.restore_state(bridge._state_init, False)
            jelly.set_node_velocities_local(np.tile(v0, num_nodes))
            recorder.begin_iteration()
            for step in range(num_steps):
                scene.step(dt)
                caption = (
                    f"iteration {iteration}   t = {(step + 1) * dt:.2f} s\n"
                    f"v0 = [{v0[0]:+.2f} {v0[1]:+.2f} {v0[2]:+.2f}] m/s   loss = {losses[-1]:.4f}"
                )
                recorder.capture(centroid_world(), caption, hold=(1 if step < num_steps - 1 else 12))
        optimizer.step()

    bridge.close()
    recorder.write(output_dir / "soft_landing.mp4")
    save_loss_curve(output_dir / "soft_landing_loss.png", losses, "Soft landing: loss vs iteration")
    physics.destroy_scene(scene)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("diffsim_videos"))
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--task", choices=["rigid", "soft", "both"], default="both")
    parser.add_argument(
        "--export-scenes",
        action="store_true",
        help="also export the recorded frames for render_diffsim_blender.py",
    )
    args = parser.parse_args()
    global EXPORT_SCENES
    EXPORT_SCENES = args.export_scenes
    args.output_dir.mkdir(parents=True, exist_ok=True)

    physics.initialize(num_worker_threads=0)
    start = time.time()
    if args.task in ("rigid", "both"):
        task_rigid_throw(args.output_dir, args.iterations)
    if args.task in ("soft", "both"):
        task_soft_landing(args.output_dir, args.iterations)
    print(f"done in {time.time() - start:.1f} s")
    physics.shutdown()


if __name__ == "__main__":
    main()
