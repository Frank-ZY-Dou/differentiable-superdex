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

Three tasks are solved by gradient-based optimization where every gradient comes
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
   descent converges monotonically here, by several orders of magnitude in
   20 iterations; Adam's per-coordinate normalization overshoots the narrow
   valley and oscillates.)
3. ``soft_on_soft.mp4`` - a jelly cube is thrown at a jelly cube resting on the
   ground and shoves it along. Gradient descent on the launch velocity, through
   the contact between the two deformable bodies and the resting cube's
   friction on the ground, brings the resting cube's centroid to a target.

Each video shows a selection of optimization iterations back to back: the
rollout of that iteration, the trail of the tracked point, the target, and the
current loss. The loss curves are also written as PNG. At the first iteration
of every task the adjoint gradient is compared with central finite differences
of the rollout (``check_gradient``).

On the soft task the engine may log ``Zero Preconditioner-dot product`` from
its PCG adjoint solve on some steps: the PSD-projected approximate Hessian
used as preconditioner can be singular on a contact island, PCG aborts and the
engine falls back to MINRES. The per-iteration ``adjoint residual`` printed by
this script (about 1e-10) confirms every adjoint solve still converged.

Requirements: double precision (selected below, before the first physics
import), ``polyscope`` (offscreen rendering), ``imageio`` with ffmpeg, OpenCV
(text overlays) and PyTorch (tasks 2-4). Run from anywhere::

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
from superdex.physics.diffsim_rollout import step_with_substeps
from superdex.physics.utils.penetration import PenetrationChecker
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


def box_tet_mesh(size, cells):
    """A structured tetrahedral mesh of a box centered at the origin: ``cells``
    hexahedra per axis (one count or three), each split into six tetrahedra
    (Kuhn triangulation), spanning ``size`` (one length or three). Returns
    (coordinates, connectivity) flattened the way ``create_tet_mesh_shape``
    expects them."""
    size = np.broadcast_to(np.asarray(size, dtype=np.float64), (3,))
    cells = np.broadcast_to(np.asarray(cells, dtype=np.int64), (3,))
    n = cells + 1
    axes = [np.linspace(-0.5 * size[a], 0.5 * size[a], n[a]) for a in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)  # (n0,n1,n2,3)
    coordinates = grid.reshape(-1, 3)

    def node(i, j, k):
        return (i * n[1] + j) * n[2] + k

    kuhn = [
        (0, 1, 3, 7),
        (0, 1, 5, 7),
        (0, 4, 5, 7),
        (0, 4, 6, 7),
        (0, 2, 6, 7),
        (0, 2, 3, 7),
    ]
    tets = []
    for i in range(cells[0]):
        for j in range(cells[1]):
            for k in range(cells[2]):
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
    the textures as PNG files); ``render_diffsim_blender.py`` renders it with Blender.
    ``target`` None marks no target; ``colors`` maps actor names to the base color of
    their physics mesh (the renderer picks one otherwise)."""

    def __init__(self, scene, target, look_from, look_at, title: str, colors=None,
                 view_transform=None, view_exposure=None):
        self.scene = scene
        self.colors = {name: [float(c) for c in color] for name, color in (colors or {}).items()}
        self.meta = {
            "title": title,
            "view_transform": view_transform,
            "view_exposure": view_exposure,
            "target": None if target is None else [float(v) for v in target],
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
            material={"base_color": self.colors.get(actor.get_name()), "metallic": None, "roughness": None},
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
            "tracked": None if tracked_point is None else [float(v) for v in np.asarray(tracked_point, dtype=np.float64)],
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
    """Offscreen viewer + MP4 writer with text overlays and a tracked-point trail
    (``target`` None: no marker; a captured ``tracked_point`` of None: no trail).
    With ``EXPORT_SCENES`` set, the recorded frames are also exported for Blender
    (see :class:`SceneExport`, which ``colors`` is passed to)."""

    def __init__(self, scene, target, look_from, look_at, title: str, curves=None, colors=None,
                 view_transform=None, view_exposure=None):
        # The scene is Z-up FLU (X forward, Y left, Z up) - the "ros" preset.
        self.viewer = Viewer(
            ViewerCfg(offscreen=True, size=FRAME_SIZE, coordinate_system="ros")
        )
        self.viewer.set_scene(scene)
        if target is not None:
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
            SceneExport(scene, target, look_from, look_at, title, colors, view_transform, view_exposure)
            if EXPORT_SCENES else None
        )

    def begin_iteration(self) -> None:
        self.trail = []
        if self.export is not None:
            self.export.begin_iteration()

    def capture(self, tracked_point, caption: str, hold: int = 1) -> None:
        if tracked_point is not None:
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


def check_gradient(name: str, loss_of, params, grad, labels, tolerance: float | None) -> float:
    """Compares the adjoint gradient ``grad`` of the rollout loss ``loss_of`` at ``params``
    (both flat) with central finite differences of the rollout, on every entry and along
    the gradient, at the step sizes 1e-5 and 1e-6. An entry validates the adjoint where
    its two quotients agree with each other to 1e-4 ("smooth"; the loss of a contact task
    is not smooth at every scale, and such entries are reported, not judged). With
    ``tolerance`` a smooth entry whose relative error exceeds it raises. Returns the
    relative error along the gradient."""
    params = np.asarray(params, dtype=np.float64).ravel()
    grad = np.asarray(grad, dtype=np.float64).ravel()
    print(f"[{name}] gradient check (adjoint vs rollout central FD at eps 1e-5 / 1e-6):")

    def quotients(direction: np.ndarray) -> list[float]:
        return [
            (loss_of(params + eps * direction) - loss_of(params - eps * direction)) / (2.0 * eps)
            for eps in (1e-5, 1e-6)
        ]

    norm = float(np.linalg.norm(grad))
    rows = [("along the gradient", norm, quotients(grad / norm))]
    for index, label in enumerate(labels):
        rows.append((label, float(grad[index]), quotients(np.eye(len(params))[index])))
    worst = 0.0
    for label, adjoint, fds in rows:
        denominator = max(abs(fds[0]), 1e-30)
        fd_self = abs(fds[0] - fds[1]) / denominator
        rel = abs(adjoint - fds[0]) / denominator
        smooth = fd_self < 1e-4
        print(
            f"    {label:>18s}: adjoint {adjoint:+.5e}  fd {fds[0]:+.5e}  rel err {rel:.1e}  "
            f"fd self-consistency {fd_self:.0e}  [{'smooth' if smooth else 'rough '}]"
        )
        if smooth:
            worst = max(worst, rel)
    if tolerance is not None and worst > tolerance:
        raise RuntimeError(
            f"[{name}] the adjoint gradient is off by {worst:.1e} relative on a smooth entry "
            f"(tolerance {tolerance:.0e})"
        )
    along = rows[0][2][0]
    return abs(norm - along) / abs(along)


def assert_converged(scene, tolerance: float) -> None:
    """After a plain ``scene.step`` of a replay: the Newton solve must have converged, or
    stalled with a residual below ``tolerance`` (the scene's round-off floor)."""
    stats = scene.get_solver_stats()
    status = stats.convergence_status
    if status == physics.ConvergenceStatus.CONVERGED:
        return
    if status == physics.ConvergenceStatus.STOPPED and stats.residual_norm <= tolerance:
        return
    raise RuntimeError(
        f"forward Newton solve did not converge (status {status}, "
        f"residual {stats.residual_norm:.2e}, iterations {stats.max_non_linear_iters})"
    )


def jelly_material():
    """The neo-Hookean jelly of the soft tasks (soft enough to squash visibly on impact)."""
    material = physics.SoftMaterialParams(density=1000.0, mass_damping_coefficient=1.0)
    material.neo_hookean = physics.NeoHookeanMaterialParams(youngs_modulus=4.0e4, poisson_ratio=0.45)
    return material


def soft_centroid(actor, rest: np.ndarray):
    """The world centroid of a soft actor's nodes (``rest``: their rest positions in the
    actor's frame) and the rotation of the actor's frame, which the centroid's
    derivative with respect to the displacements is 1/N times."""
    positions = rest + np.asarray(actor.get_displacements(), dtype=np.float64).reshape(-1, 3)
    if not actor.has_root_transform():
        return positions.mean(axis=0), np.eye(3)
    position, rotvec = transformrt_to_numpy(actor.get_root_transform())
    world_from_local = np.asarray(make_transform(position, rotvec), dtype=np.float64)
    return world_from_local[:3, :3] @ positions.mean(axis=0) + world_from_local[:3, 3], world_from_local[:3, :3]


SOFT_MAX_PENETRATION = 0.02  # [m] no contact sample deeper than a tenth of the jelly's side


class SoftMonitor:
    """Contact overlap and element validity of the soft bodies, observed on every accepted
    (sub)step of every rollout - the optimizer's, the finite-difference probes', the camera
    replays' - and judged after each rollout by an explicit policy:

    - no contact sample of any actor pair deeper than ``max_penetration``. The measure is
      the engine's own contact samples (``PenetrationChecker``): a sampled coverage of
      the penalty contact's overlap, not a collision certificate - a vertex between two
      samples, or a thin body crossing between samples within a substep, is not seen.
      The bound is a tenth of the jelly's side: deeper, the overlap shows in the render
      and the penalty contact no longer stands in for a non-penetrating one;
    - no inverted tetrahedron: the determinant of every element's deformation gradient
      (its deformed over its rest signed volume) stays positive, in the initial state
      (checked at construction) and after every accepted (sub)step.

    The forward solve's convergence is the rollout driver's separate condition
    (``substep_residual_tolerance``). ``begin()`` opens a rollout and forgets the previous
    one's records (a restored state carries no record), ``observe`` is the driver's
    ``observe_substep`` hook and ``step`` its stand-in for a replay outside the bridge,
    ``check`` applies the policy. ``bodies`` maps each soft actor to its rest coordinates
    and tetrahedra."""

    def __init__(self, scene, bodies: dict, max_penetration: float):
        self.scene = scene
        self.penetration = PenetrationChecker(scene)
        self.max_penetration = float(max_penetration)
        self.bodies = []
        for actor, (coordinates, connectivity) in bodies.items():
            rest = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
            tets = np.asarray(connectivity, dtype=np.int64).reshape(-1, 4)
            rest_det = np.linalg.det(rest[tets[:, 1:]] - rest[tets[:, :1]])
            if not np.all(rest_det > 0.0):
                raise ValueError(f"{actor.get_name()}: {int((rest_det <= 0.0).sum())} elements of non-positive rest volume")
            self.bodies.append((actor, rest, tets, rest_det))
        self.min_det = float("inf")
        self.num_observed = 0
        self.rollouts = 0
        self.observed_total = 0
        self.worst_depth = 0.0
        self.worst_det = float("inf")
        initial = self._min_det()
        if not initial > 0.0:
            raise RuntimeError(f"inverted element in the initial state (min det {initial:.3e})")

    def _min_det(self) -> float:
        value = float("inf")
        for actor, rest, tets, rest_det in self.bodies:
            positions = rest + np.asarray(actor.get_displacements(), dtype=np.float64).reshape(-1, 3)
            det = np.linalg.det(positions[tets[:, 1:]] - positions[tets[:, :1]]) / rest_det
            value = min(value, float(det.min()))
        return value

    def begin(self) -> None:
        self.penetration.reset()
        self.min_det = float("inf")
        self.num_observed = 0

    def observe(self, step: int, sub_dt: float) -> None:
        self.penetration.record(step)
        self.min_det = min(self.min_det, self._min_det())
        self.num_observed += 1

    def step(self, dt: float, step: int) -> None:
        """A replay step outside the bridge, substepped like the rollout's, every accepted piece observed."""

        def on_substep(pre, post, sub_dt: float) -> None:
            self.scene.release_state(pre)
            self.scene.release_state(post)
            self.observe(step, sub_dt)

        step_with_substeps(self.scene, dt, SOFT_SUBSTEPS, SOFT_RESIDUAL_TOLERANCE, step=step, on_substep=on_substep)

    def check(self, label: str) -> None:
        """The policy on the rollout observed since ``begin()``; raises on a violation."""
        if self.num_observed == 0:
            raise RuntimeError(f"{label}: no (sub)step was observed")
        if not self.min_det > 0.0:
            raise RuntimeError(f"{label}: inverted element (min deformation-gradient determinant {self.min_det:.3e})")
        try:
            self.penetration.assert_below(self.max_penetration)
        except RuntimeError as error:
            raise RuntimeError(f"{label}: {error}") from error
        self.rollouts += 1
        self.observed_total += self.num_observed
        self.worst_depth = max(self.worst_depth, self.penetration.max_depth())
        self.worst_det = min(self.worst_det, self.min_det)

    def summary(self) -> str:
        return (
            f"{self.rollouts} rollouts, {self.observed_total} accepted (sub)steps observed: deepest contact "
            f"sample {1000.0 * self.worst_depth:.2f} mm (limit {1000.0 * self.max_penetration:.0f} mm, sampled "
            f"coverage), min element determinant {self.worst_det:.3f} (limit > 0)"
        )


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
    coordinates, connectivity = box_tet_mesh(size=0.2, cells=5)
    num_nodes = coordinates.size // 3

    scene = physics.create_scene("Differentiable soft landing")
    scene.set_gravity(GRAVITY)
    contact = physics.ContactParams(penalty_coefficient=1e8, coulomb_friction_coefficient=0.4)
    scene.create_rigid_actor(
        name="ground",
        shape=physics.create_plane_shape(normal=[0, 0, 1], distance=0.0),
        is_static=True,
        contact=contact,
    )
    jelly = scene.create_soft_actor(
        name="jelly",
        shape=physics.create_tet_mesh_shape(coordinates=coordinates, connectivity=connectivity),
        material=jelly_material(),
        contact=contact,
        world_from_local=physics.TransformRT([0.0, 0.0, 0.35]),
    )
    diffsim.make_scene_differentiable(scene)  # also disables recentering (fixed root frame)
    tighten_solvers(scene)
    monitor = SoftMonitor(scene, {jelly: (coordinates, connectivity)}, SOFT_MAX_PENETRATION)
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
        view_transform="Standard",
        view_exposure=-1.2,
    )
    bridge = TorchRollout(
        scene,
        dt=dt,
        num_steps=num_steps,
        initial_state_actors=[jelly],
        terminal_losses=[CentroidLoss()],
        max_substep_levels=SOFT_SUBSTEPS,
        substep_residual_tolerance=SOFT_RESIDUAL_TOLERANCE,
        observe_substep=monitor.observe,
    )
    velocity = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([velocity], lr=4.0)
    losses = []
    record = recorded_iterations(num_iterations)
    u0 = torch.zeros(3 * num_nodes, dtype=torch.float64)

    def rollout(launch: torch.Tensor) -> torch.Tensor:
        monitor.begin()
        loss = bridge(initial_states=torch.cat([u0, launch.repeat(num_nodes)]))
        monitor.check("rollout")
        return loss

    for iteration in range(num_iterations):
        optimizer.zero_grad()
        loss = rollout(velocity)
        loss.backward()
        losses.append(float(loss))
        v0 = velocity.detach().numpy().copy()
        result = bridge.last_result
        if iteration == 0:
            fd_error = check_gradient(
                "soft",
                lambda values: float(rollout(torch.as_tensor(values, dtype=torch.float64)).detach()),
                v0,
                velocity.grad.numpy(),
                ("v0 x", "v0 y", "v0 z"),
                tolerance=None,
            )
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
            monitor.begin()
            for step in range(num_steps):
                monitor.step(dt, step)
                caption = (
                    f"iteration {iteration}   t = {(step + 1) * dt:.2f} s\n"
                    f"v0 = [{v0[0]:+.2f} {v0[1]:+.2f} {v0[2]:+.2f}] m/s   loss = {losses[-1]:.4f}"
                )
                recorder.capture(centroid_world(), caption, hold=(1 if step < num_steps - 1 else 12))
            monitor.check(f"replay of iteration {iteration}")
        optimizer.step()

    print(
        f"[soft] loss {losses[0]:.3e} -> {losses[-1]:.3e} in {num_iterations} iterations, "
        f"adjoint vs FD along the gradient {fd_error:.1e}"
    )
    print("[soft] " + monitor.summary())
    print("[soft] last replay: " + monitor.penetration.report(SOFT_MAX_PENETRATION).replace("\n", "\n[soft] "))
    bridge.close()
    recorder.write(output_dir / "soft_landing.mp4")
    save_loss_curve(output_dir / "soft_landing_loss.png", losses, "Soft landing: loss vs iteration")
    physics.destroy_scene(scene)


# ---------------------------------------------------------------------------
# Task 3: a thrown jelly shoves a resting jelly to a target (soft-soft contact)
# ---------------------------------------------------------------------------

JELLY_SIDE, JELLY_CELLS = 0.2, 4
SOFT_ON_SOFT_DT, SOFT_ON_SOFT_STEPS = 0.02, 50  # 1 s
SOFT_ON_SOFT_SETTLE_STEPS = 40  # the resting jelly settles on the ground before the other is added
SOFT_ON_SOFT_LAUNCH = [-0.42, 0.0, 0.14]  # [m] the thrown jelly's center at launch
SOFT_ON_SOFT_V0 = [1.5, 0.0, 0.0]  # [m/s] the first guess
SOFT_ON_SOFT_TARGET = [0.12, 0.03]  # [m] the resting jelly's centroid (x, y) at the end
SOFT_ON_SOFT_STEP, SOFT_ON_SOFT_DECAY = 0.15, 0.9  # normalized descent: step [m/s], decay per iteration
SOFT_RESIDUAL_TOLERANCE = 1e-4  # [N] the soft-contact Newton solve floors near 1.3e-5 (measured); this bounds a blow-up
SOFT_SUBSTEPS = 4   # a hard impact step (residual above the tolerance) is redone as 2, 4, 8, 16 substeps


def task_soft_on_soft(output_dir: pathlib.Path, num_iterations: int) -> None:
    import torch
    from superdex.physics.diffsim_torch import TorchRollout

    coordinates, connectivity = box_tet_mesh(JELLY_SIDE, JELLY_CELLS)
    rest = coordinates.reshape(-1, 3)
    num_nodes = len(rest)
    target = np.array(SOFT_ON_SOFT_TARGET + [0.5 * JELLY_SIDE])  # the marker at the jelly's height

    scene = physics.create_scene("Differentiable soft on soft")
    scene.set_gravity(GRAVITY)
    contact = physics.ContactParams(penalty_coefficient=1e8, coulomb_friction_coefficient=0.4)
    scene.create_rigid_actor(
        name="ground",
        shape=physics.create_plane_shape(normal=[0, 0, 1], distance=0.0),
        is_static=True,
        contact=contact,
    )

    def jelly(name: str, position):
        # Soft actors are no colliders by default; created through the experimental API each
        # jelly is an SDF collider too, so that the two test their contact samples against
        # each other (the deformed rest-space grid of the other).
        experimental = physics.experimental
        return experimental.create_soft_actor(
            scene,
            physics.SoftActorParams(
                name=name,
                shape=physics.create_tet_mesh_shape(coordinates=coordinates, connectivity=connectivity),
                material=jelly_material(),
                contact=contact,
                world_from_local=physics.TransformRT(list(position)),
            ),
            experimental.ExperimentalSoftActorParams(collider_type=physics.ColliderType.SDF),
        )

    resting = jelly("resting", [0.0, 0.0, 0.5 * JELLY_SIDE])
    for _ in range(SOFT_ON_SOFT_SETTLE_STEPS):  # about 1 cm under its own weight
        scene.step(SOFT_ON_SOFT_DT)
    thrown = jelly("thrown", SOFT_ON_SOFT_LAUNCH)
    diffsim.make_scene_differentiable(scene)  # also disables recentering (fixed root frames)
    tighten_solvers(scene)
    monitor = SoftMonitor(
        scene, {resting: (coordinates, connectivity), thrown: (coordinates, connectivity)}, SOFT_MAX_PENETRATION
    )

    class ShoveLoss:
        """0.5 |c_xy - target_xy|^2 on the resting jelly's centroid at the last step (its
        height is the settled one, which the launch cannot change)."""

        def residual(self):
            centroid, rotation = soft_centroid(resting, rest)
            return centroid[:2] - target[:2], rotation

        def value(self) -> float:
            d, _ = self.residual()
            return 0.5 * float(d @ d)

        def accumulate_output_grad(self) -> None:
            d, rotation = self.residual()
            grad_world = np.array([d[0], d[1], 0.0]) / num_nodes
            diffsim.get_displacements_backward(resting, np.tile(rotation.T @ grad_world, num_nodes))

    recorder = Recorder(
        scene,
        target,
        look_from=[-0.35, -1.6, 0.8],
        look_at=[-0.05, 0.0, 0.12],
        title="Soft on soft: gradient descent on the launch velocity (soft-soft contact adjoint)",
        colors={"thrown": (0.75, 0.33, 0.05), "resting": (0.06, 0.42, 0.14)},
        view_transform="Standard",
        view_exposure=-1.2,
    )
    bridge = TorchRollout(
        scene,
        dt=SOFT_ON_SOFT_DT,
        num_steps=SOFT_ON_SOFT_STEPS,
        initial_state_actors=[thrown],
        terminal_losses=[ShoveLoss()],
        max_substep_levels=SOFT_SUBSTEPS,
        substep_residual_tolerance=SOFT_RESIDUAL_TOLERANCE,
        observe_substep=monitor.observe,
    )
    u0 = torch.zeros(3 * num_nodes, dtype=torch.float64)

    def rollout(launch: torch.Tensor) -> torch.Tensor:
        monitor.begin()
        loss = bridge(initial_states=torch.cat([u0, launch.repeat(num_nodes)]))
        monitor.check("rollout")
        return loss

    def replay(v0: np.ndarray, iteration: int) -> None:
        scene.restore_state(bridge._state_init, False)
        thrown.set_node_velocities_local(np.tile(v0, num_nodes))
        recorder.begin_iteration()
        monitor.begin()
        for step in range(SOFT_ON_SOFT_STEPS):
            monitor.step(SOFT_ON_SOFT_DT, step)
            caption = (
                f"iteration {iteration}   t = {(step + 1) * SOFT_ON_SOFT_DT:.2f} s\n"
                f"v0 = [{v0[0]:+.2f} {v0[1]:+.2f} {v0[2]:+.2f}] m/s   loss = {losses[-1]:.4f}"
            )
            recorder.capture(
                soft_centroid(resting, rest)[0], caption, hold=(1 if step < SOFT_ON_SOFT_STEPS - 1 else 12)
            )
        monitor.check(f"replay of iteration {iteration}")

    velocity = torch.tensor(SOFT_ON_SOFT_V0, dtype=torch.float64, requires_grad=True)
    losses = []
    record = recorded_iterations(num_iterations)
    for iteration in range(num_iterations):
        if velocity.grad is not None:
            velocity.grad.zero_()
        loss = rollout(velocity)
        loss.backward()
        losses.append(float(loss))
        v0 = velocity.detach().numpy().copy()
        result = bridge.last_result
        if iteration == 0:
            fd_error = check_gradient(
                "soft_on_soft",
                lambda values: float(rollout(torch.as_tensor(values, dtype=torch.float64)).detach()),
                v0,
                velocity.grad.numpy(),
                ("v0 x", "v0 y", "v0 z"),
                tolerance=None,
            )
        print(
            f"[soft_on_soft] iter {iteration:3d}  loss {losses[-1]:.6f}  v0 {v0.round(3)}  "
            f"|grad| {float(velocity.grad.norm()):.3e}  adjoint residual {result.max_adjoint_residual:.1e}"
        )
        if iteration in record:
            replay(v0, iteration)
        with torch.no_grad():  # a step of decaying length along the gradient's direction
            step = SOFT_ON_SOFT_STEP * SOFT_ON_SOFT_DECAY**iteration
            velocity -= step / (velocity.grad.norm() + 1e-30) * velocity.grad

    print(
        f"[soft_on_soft] loss {losses[0]:.3e} -> {losses[-1]:.3e} in {num_iterations} iterations, "
        f"adjoint vs FD along the gradient {fd_error:.1e}"
    )
    print("[soft_on_soft] " + monitor.summary())
    print(
        "[soft_on_soft] last replay: "
        + monitor.penetration.report(SOFT_MAX_PENETRATION).replace("\n", "\n[soft_on_soft] ")
    )
    bridge.close()
    recorder.write(output_dir / "soft_on_soft.mp4")
    save_loss_curve(output_dir / "soft_on_soft_loss.png", losses, "Soft on soft: loss vs iteration")
    physics.destroy_scene(scene)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("diffsim_videos"))
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--task", choices=["rigid", "soft", "soft_on_soft", "all"], default="all")
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
    if args.task in ("rigid", "all"):
        task_rigid_throw(args.output_dir, args.iterations)
    if args.task in ("soft", "all"):
        task_soft_landing(args.output_dir, args.iterations)
    if args.task in ("soft_on_soft", "all"):
        task_soft_on_soft(args.output_dir, args.iterations)
    print(f"done in {time.time() - start:.1f} s")
    physics.shutdown()


if __name__ == "__main__":
    main()
