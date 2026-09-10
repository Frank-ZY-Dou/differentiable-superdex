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

"""Differentiable tactile control of the XHand1 in SuperDex (double precision).

The hand grasps a cylinder and reads a tactile image: for each of the five
fingertips, the contact forces on the pad are distributed over a 10 x 12 grid of
taxels, so the hand reports a 5 x 120 compression map (600 taxels). The map is a
differentiable function of the simulator state, and the demo optimises the twelve
joint targets of the hand's pose controller through it.

Two tasks:

- ``--task force`` regulates the total force on each pad to F_REF newtons. The
  signal is the engine's total contact force per pad; the gradient flows through
  the joint servo and the frictional contact by ``diffsim.get_contact_force_world_backward``.

- ``--task field`` (the default) matches the whole 600-taxel compression map to a
  target map (the map of the 3 N grip). The readout is written as a torch
  function of the engine's per-contact forces and the fingertip rotations, and its
  gradient reaches the joint targets through the engine's per-contact backward
  ``diffsim.get_contact_points_backward`` and ``diffsim.get_root_transform_backward``.
  The target is the 3 N grip the force task settles into; starting from the ~12 N
  settled grip, the demo recovers that grip's twelve joint targets using only the
  tactile map's gradient (the readout is validated against finite differences to
  about 1e-7).

Both tasks check the adjoint gradient against central finite differences of the
same rollout at the first iteration (``--check``). The optimisation is rendered
with Blender (the hand grasping the cylinder in the same style as the other demos)
beside the current and target taxel maps:

    SUPERDEX_PRECISION=double python example_diffsim_tactile.py --task field --check \
        --iters 70 --output-dir diffsim_videos
    SUPERDEX_PRECISION=double python example_diffsim_tactile.py \
        --render diffsim_videos/xhand_tactile_field.npz --output-dir diffsim_videos

The hand package (``assets/bots/hands/xhand1_official``) is RobotEra's XHand1 v1.3;
the taxel layout is reproduced from https://github.com/tsingqingyun/xhand1 (see that
package's ``tactile/README.md``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

os.environ.setdefault("SUPERDEX_PRECISION", "double")

import numpy as np
import superdex.physics as physics
import superdex.robotics as robotics
import torch
from superdex.physics import diffsim
from superdex.physics.diffsim_torch import TorchRollout
from superdex.physics.paths import resolve_asset
from superdex.physics.utils import render_model_registry

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import example_diffsim_video as video  # noqa: E402  (SceneExport, Recorder, save_loss_curve)

# The XHand1 v1.3 bot package and the taxel layout registered against its meshes.
BOT = "bots/hands/xhand1_official/{side}/xhand1_v13_{side}.superdex_bot"
PACKAGE = resolve_asset(BOT.format(side="right")).parent.parent
MESH_DIR = PACKAGE / "meshes"
TACTILE_DIR = PACKAGE / "tactile"

# ---------------------------------------------------------------------------
# Taxel readout
# ---------------------------------------------------------------------------

FINGERS = ("thumb", "index", "mid", "ring", "pinky")
# distal link of each finger (the pad the taxels sit on)
LINK2 = {"thumb": "thumb_rota_link2", "index": "index_rota_link2", "mid": "mid_link2",
         "ring": "ring_link2", "pinky": "pinky_link2"}
# finger axis in the link frame (base -> tip)
ALONG = {"thumb": np.array([0.0, 1.0, 0.0]), "index": np.array([0.0, 0.0, 1.0]),
         "mid": np.array([0.0, 0.0, 1.0]), "ring": np.array([0.0, 0.0, 1.0]),
         "pinky": np.array([0.0, 0.0, 1.0])}
ROWS, COLS = 10, 12                    # point-1 = 12 * row + col; col runs base -> tip
RADIUS = 0.005                         # taxel kernel support [m] (pitch 2 mm, hull-vs-skin offset up to 2 mm)
COS_MIN = 0.3                          # a taxel takes a contact only if its outward normal faces it
SOURCE_SHA256 = {"t16.json": "fc58f8da95f0a1ab", "t30_left.json": "3faad15a4845438c",
                 "t30_right.json": "d816237de7543a1f"}


def wendland(r):
    """Compact C2 kernel on r = d / RADIUS (zero for r >= 1)."""
    r = np.clip(r, 0.0, 1.0)
    return (1.0 - r) ** 4 * (4.0 * r + 1.0)


def nearest_faces(mesh, points, search=0.004):
    """Distance to the mesh surface and the nearest face for each point.

    Candidate faces are those whose centroid lies within ``search`` of the point
    (the taxels sit on the surface, so the true nearest face is among them)."""
    import trimesh

    centers = mesh.triangles_center
    dist = np.empty(len(points))
    face = np.empty(len(points), int)
    for i, p in enumerate(points):
        cand = np.flatnonzero(np.linalg.norm(centers - p, axis=1) < search)
        if not len(cand):
            raise ValueError(f"no mesh face within {search * 1e3:.0f} mm of taxel {i}")
        closest = trimesh.triangles.closest_point(mesh.triangles[cand], np.repeat(p[None], len(cand), 0))
        d = np.linalg.norm(closest - p, axis=1)
        j = int(np.argmin(d))
        dist[i], face[i] = d[j], cand[j]
    return dist, face


class Layout:
    """Taxel positions, outward normals and tangent bases of one hand, per finger, in link frames (m)."""

    def __init__(self, side):
        import trimesh

        if side not in ("left", "right"):
            raise ValueError(side)
        self.side = side
        self.pos = np.zeros((5, 120, 3))
        self.normal = np.zeros((5, 120, 3))
        self.t_along = np.zeros((5, 120, 3))
        self.t_across = np.zeros((5, 120, 3))
        self.surface_dist = np.zeros((5, 120))
        for fi, finger in enumerate(FINGERS):
            fname = f"t30_{side}.json" if finger == "thumb" else "t16.json"
            raw_bytes = (TACTILE_DIR / fname).read_bytes()
            digest = hashlib.sha256(raw_bytes).hexdigest()[:16]
            if digest != SOURCE_SHA256[fname]:
                raise ValueError(f"{fname}: sha256 {digest} is not the recorded layout {SOURCE_SHA256[fname]}")
            raw = json.loads(raw_bytes)
            if raw["sensor_model"] != ("T30" if finger == "thumb" else "T16"):
                raise ValueError(f"{fname}: sensor_model {raw['sensor_model']}")
            if finger == "thumb" and raw["hand"] != side:
                raise ValueError(f"{fname}: hand {raw['hand']} != {side}")
            if not raw["coordinate_system"].startswith("transformed"):
                raise ValueError(f"{fname}: expected already-transformed coordinates")
            rows = sorted(raw["measurement_points"], key=lambda p: p["point"])
            if [p["point"] for p in rows] != list(range(1, 121)):
                raise ValueError(f"{fname}: point ids are not 1..120")
            self.pos[fi] = np.array([[p["x"], p["y"], p["z"]] for p in rows]) * 1e-3  # mm -> m
            # outward normal from the distal-link mesh (winding-consistent, watertight)
            mesh = trimesh.load(MESH_DIR / f"{side}_hand_{LINK2[finger]}.STL", force="mesh")
            if not (mesh.is_winding_consistent and mesh.is_watertight):
                raise ValueError(f"{finger}: mesh is not a closed, consistently oriented surface")
            dist, tri = nearest_faces(mesh, self.pos[fi])
            self.surface_dist[fi] = dist
            n = mesh.face_normals[tri].copy()
            # smooth the triangle-noisy normals over the nearest taxels (weighted by distance)
            d = np.linalg.norm(self.pos[fi][:, None] - self.pos[fi][None], axis=2)
            w = wendland(d / 0.004)
            n = (w[:, :, None] * n[None]).sum(1)
            n /= np.linalg.norm(n, axis=1, keepdims=True)
            along = ALONG[finger] - np.einsum("ij,j->i", n, ALONG[finger])[:, None] * n
            along /= np.linalg.norm(along, axis=1, keepdims=True)
            self.normal[fi] = n
            self.t_along[fi] = along
            self.t_across[fi] = np.cross(n, along)
        if self.surface_dist.max() > 0.002:
            raise ValueError(f"taxel layout does not sit on the meshes (max {self.surface_dist.max() * 1e3:.1f} mm)")

    @staticmethod
    def grid(force):
        """(5, 120, ...) -> (5, ROWS, COLS, ...): rows wrap around the pad, columns run base -> tip."""
        return force.reshape((5, ROWS, COLS) + force.shape[2:])


class TactileFrame:
    """One readout: forces ON the finger in the distal-link frames, newtons."""

    def __init__(self):
        self.force = np.zeros((5, 120, 3))       # taxel force, link frame
        self.normal = np.zeros((5, 120))         # compression along the outward normal (>= 0)
        self.shear = np.zeros((5, 120, 2))       # (along the finger, across the pad)
        self.covered_force = np.zeros((5, 3))    # sum of the contact forces that reached taxels (link frame)
        self.unmapped_force = np.zeros((5, 3))   # contacts on the link with no facing taxel (link frame)
        self.self_force = np.zeros((5, 3))       # contacts with the hand's own bodies (ignored, link frame)
        self.moment_residual = np.zeros((5, 3))  # about the link origin, from moving forces to taxels
        self.n_contacts = np.zeros(5, int)
        self.n_unmapped = np.zeros(5, int)

    def finish(self, layout):
        self.normal = np.maximum(0.0, -np.einsum("fij,fij->fi", self.force, layout.normal))
        self.shear[..., 0] = np.einsum("fij,fij->fi", self.force, layout.t_along)
        self.shear[..., 1] = np.einsum("fij,fij->fi", self.force, layout.t_across)
        return self


def project_finger(layout, fi, R, o, pos_w, f_w, out_w, radius=RADIUS, cos_min=COS_MIN):
    """Partition the contact forces on finger ``fi`` over its taxels.

    R (3x3, link -> world), o (3,) link origin; pos_w (N,3) contact positions,
    f_w (N,3) forces ON the finger, out_w (N,3) unit directions from the finger
    toward the other body (all world frame). Each contact is spread over the taxels
    within ``radius`` whose outward normal faces it (dot >= cos_min), weights
    normalised per contact, so the taxel forces sum to the covered contact forces.
    Returns (force (120,3) link frame, covered (N,) mask, unmapped force (3,) link
    frame, moment residual (3,) about the link origin)."""
    pos_w, f_w, out_w = (np.asarray(a, float).reshape(-1, 3) for a in (pos_w, f_w, out_w))
    p = (pos_w - o) @ R
    f = f_w @ R
    out = out_w @ R
    d = np.linalg.norm(p[:, None, :] - layout.pos[fi][None], axis=2)      # (N, 120)
    facing = (out @ layout.normal[fi].T) >= cos_min
    k = wendland(d / radius) * facing
    s = k.sum(1)
    covered = s > 0.0
    w = np.zeros_like(k)
    w[covered] = k[covered] / s[covered, None]
    force = w.T @ f
    unmapped = f[~covered].sum(0)
    moment = np.cross(p[covered], f[covered]).sum(0) - np.cross(layout.pos[fi], force).sum(0)
    return force, covered, unmapped, moment


def heat(v):
    """0..1 -> RGB (black -> red -> yellow -> white), uint8."""
    v = np.clip(np.asarray(v, float), 0.0, 1.0)
    r = np.clip(v * 3.0, 0, 1)
    g = np.clip(v * 3.0 - 1.0, 0, 1)
    b = np.clip(v * 3.0 - 2.0, 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def panel(frames, scale, height, width=380, cell=7, labels=("current", "target")):
    """A side panel: for each frame a column of the five 10 x 12 taxel grids (columns base -> tip)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (24, 24, 28))
    dr = ImageDraw.Draw(img)
    colw = width // len(frames)
    for h, (fr, lab) in enumerate(zip(frames, labels)):
        x0 = h * colw + 12
        dr.text((x0, 6), f"{lab}   max {fr.normal.max():.2f} N", fill=(230, 230, 230))
        y = 26
        for fi, finger in enumerate(FINGERS):
            grid = Layout.grid(fr.normal)[fi]
            tile = heat(grid / scale)
            tile = np.repeat(np.repeat(tile, cell, 0), cell, 1)
            img.paste(Image.fromarray(tile), (x0, y + 12))
            dr.text((x0, y), f"{finger:5s} {fr.normal[fi].sum():5.2f} N  {int((fr.normal[fi] > 0.01).sum()):3d} taxels",
                    fill=(200, 200, 200))
            y += ROWS * cell + 12 + 8
    return np.asarray(img)


# ---------------------------------------------------------------------------
# The hand rig
# ---------------------------------------------------------------------------

CHILD = {"thumb": ("thumb_rotaback_link2", "thumb_rota_tip"), "index": ("index_rotaback_link2", "index_rota_tip"),
         "mid": ("midback_link2", "mid_tip"), "ring": ("ringback_link2", "ring_tip"),
         "pinky": ("pinkyback_link2", "pinky_tip")}      # nested actors welded to each distal link
CONTACT = physics.ContactParams(penalty_coefficient=1e9, coulomb_friction_coefficient=0.8)
# Joint servo PD gains (kp 100 N.m/rad, kd 6 N.m.s/rad) and rotor armature (0.05 kg.m^2
# per joint). SuperDex's PoseTrackingParams.saturation is an angle, and the elastic torque
# saturates at stiffness * saturation, so saturation = effort_limit / kp caps the torque at
# the URDF effort limit, which matches the hand's rated ~12 N fingertip force.
GAIN = (100.0, 6.0)
ARMATURE = 0.05
DT = 0.005
_initialised = False


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Rig:
    """One hand (root welded to the world at the origin: fingers +z, palm +x, thumb +y), a
    joint pose controller, and contact queries on the five distal links."""

    def __init__(self, side="right", name="xhand tactile"):
        global _initialised
        if not _initialised:
            physics.initialize(num_worker_threads=0)
            _initialised = True
        self.side = side
        self.scene = physics.create_scene(name)
        self.scene.set_gravity([0.0, 0.0, -9.81])
        prefab = robotics.load_bot_prefab_from_file(str(resolve_asset(BOT.format(side=side))))
        j0 = prefab.joints[0]
        j0.type = physics.ArticulatedJointType.HARD  # weld the root to the world
        prefab.joints[0] = j0
        for i in range(len(prefab.links)):
            link = prefab.links[i]
            link.contact = CONTACT
            prefab.links[i] = link
        # Disable contact inside each finger chain: the welded back covers wrap around the
        # joints and overlap the neighbouring phalanges. Fingers still collide with each
        # other and with objects.
        names = [prefab.links[i].name for i in range(len(prefab.links))]
        for finger in FINGERS:
            chain = [n for n in names if finger in n.split("_hand_", 1)[1]]
            for i, a in enumerate(chain):
                for b in chain[i + 1:]:
                    ov = robotics.BotContactOverride()
                    ov.link_a, ov.link_b, ov.enable = a, b, False
                    prefab.contact_overrides.append(ov)
        self.context = robotics.create_context()
        self.bot = robotics.create_bot(self.scene, prefab, self.context)
        self.arm = self.bot.get_articulated_actor()
        self.joint_names = [prefab.joints[i].name for i in range(len(prefab.joints))
                            if prefab.joints[i].type == physics.ArticulatedJointType.REVOLUTE]
        if len(self.joint_names) != self.arm.get_num_dofs():
            raise RuntimeError(f"{len(self.joint_names)} revolute joints but {self.arm.get_num_dofs()} dofs")
        tracking = []
        for i in range(len(prefab.joints)):
            j = prefab.joints[i]
            if j.type == physics.ArticulatedJointType.REVOLUTE:
                if not (0.0 < j.effort_limit < 10.0):
                    raise RuntimeError(f"{j.name}: effort limit {j.effort_limit} from the bot file")
                tracking.append(physics.PoseTrackingParams(stiffness=GAIN[0], damping=GAIN[1],
                                                           saturation=j.effort_limit / GAIN[0]))
            else:
                tracking.append(physics.PoseTrackingParams(stiffness=0.0, damping=0.0, saturation=-1.0))
        self.arm.add_articulated_pose_controller(physics.PoseControllerParams(joint_tracking=tracking))
        inertia = np.array(self.arm.get_articulated_joint_inertia_params().tolist())
        for i in range(len(prefab.joints)):
            if prefab.joints[i].type == physics.ArticulatedJointType.REVOLUTE:
                inertia[i] = ARMATURE
        self.arm.set_articulated_joint_inertia_params(inertia.tolist())
        self.links = {}
        for handle, link in zip(self.arm.get_nested_link_actors(), self.bot.get_bot_prefab().links):
            self.links[link.name] = self.scene.get_actor(handle)
        self.hand_handles = {a.get_handle() for a in self.links.values()}
        self.prefab_links = self.bot.get_bot_prefab().links
        self.layout = Layout(side)
        self.distal = {f: self.links[f"{side}_hand_{LINK2[f]}"] for f in FINGERS}
        self.parts = {f: [self.distal[f]] + [self.links[f"{side}_hand_{c}"] for c in CHILD[f]] for f in FINGERS}
        for f in FINGERS:
            for a in self.parts[f]:
                a.register_query(physics.QueryType.CONTACT_POINTS)
                a.register_query(physics.QueryType.TOTAL_CONTACT_FORCE)
        self.target = np.zeros(self.arm.get_num_dofs())
        self.objects = []

    def add_mesh(self, name, vertices, faces, position, static=True):
        shape = physics.create_tri_mesh_shape(coordinates=np.asarray(vertices, float).ravel().tolist(),
                                              connectivity=np.asarray(faces, np.int64).ravel().tolist())
        a = self.scene.create_rigid_actor(name=name, shape=shape, is_static=static, contact=CONTACT,
                                          world_from_local=physics.TransformRT(physics.Real3(list(map(float, position)))))
        self.objects.append(a)
        return a

    def set_targets(self, values):
        """values: {joint name suffix: rad}; joints not named keep their current target."""
        for k, v in values.items():
            hits = [i for i, n in enumerate(self.joint_names) if n.endswith(k)]
            if len(hits) != 1:
                raise KeyError(f"{k}: {len(hits)} joints match")
            self.target[hits[0]] = float(v)
        self.arm.set_articulated_target_pose(self.target)

    def pose(self):
        q = np.zeros(self.arm.get_num_dofs())
        self.arm.get_articulated_pose(q)
        return q

    def step(self, n=1):
        for _ in range(n):
            self.scene.step(DT)

    def link_frame(self, finger):
        t = self.distal[finger].get_root_transform()
        return quat_to_mat(t.rotation.tolist()), np.array(t.translation.tolist())

    def taxel_world(self, fi):
        R, o = self.link_frame(FINGERS[fi])
        return o + self.layout.pos[fi] @ R.T, self.layout.normal[fi] @ R.T

    def read(self):
        """A TactileFrame plus the engine's total contact force per finger (world, N)."""
        fr = TactileFrame()
        totals = np.zeros((5, 3))
        for fi, finger in enumerate(FINGERS):
            R, o = self.link_frame(finger)
            pos, f, out = [], [], []
            for actor in self.parts[finger]:
                h = actor.get_handle()
                totals[fi] += np.array(actor.get_contact_force_world().tolist())
                for cp in actor.get_contact_points_world():
                    if cp.actor_a == h:
                        other, p, fw, ow = cp.actor_b, cp.pos_a, np.array(cp.force.tolist()), -np.array(cp.normal.tolist())
                    else:
                        other, p, fw, ow = cp.actor_a, cp.pos_b, -np.array(cp.force.tolist()), np.array(cp.normal.tolist())
                    if other in self.hand_handles:
                        fr.self_force[fi] += fw @ R
                        continue
                    pos.append(p.tolist())
                    f.append(fw)
                    out.append(ow)
            if not pos:
                continue
            f_w = np.array(f)
            force, covered, unmapped, moment = project_finger(self.layout, fi, R, o, np.array(pos), f_w, np.array(out))
            fr.force[fi] = force
            fr.covered_force[fi] = (f_w[covered] @ R).sum(0)
            fr.unmapped_force[fi] = unmapped
            fr.moment_residual[fi] = moment
            fr.n_contacts[fi] = len(pos)
            fr.n_unmapped[fi] = int((~covered).sum())
        return fr.finish(self.layout), totals


def cylinder(radius, length):
    """A cylinder with its axis along y (across the four fingers)."""
    import trimesh

    m = trimesh.creation.cylinder(radius=radius, height=length, sections=64)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    return m.vertices, m.faces


# A five-finger wrap of a static cylinder (radius 35 mm, axis across the fingers, centred 5 mm
# toward the pinky so the four fingers straddle it). Each finger flexes on joint1 and joint2;
# the thumb bends and rotates onto the near side. With the effort cap every pad presses.
GRASP = dict(shape=("cylinder", 0.035, 0.14), pos=(0.065, -0.005, 0.125),
             fingers={"joint1": 0.8, "joint2": 1.2},
             thumb={"thumb_bend_joint": 1.0, "thumb_rota_joint1": 1.3, "thumb_rota_joint2": 1.4})


def make_object(cfg):
    kind, *dims = cfg["shape"]
    if kind != "cylinder":
        raise ValueError(kind)
    return cylinder(*dims)


def close_targets(cfg, a=1.0):
    """The closing joint targets at fraction ``a`` of the grip."""
    t = {}
    for f in ("index", "mid", "ring", "pinky"):
        for j, v in cfg["fingers"].items():
            t[f"{f}_{j}"] = a * v
    for j, v in cfg["thumb"].items():
        t[j] = a * v
    return t


# ---------------------------------------------------------------------------
# The differentiable tactile tasks
# ---------------------------------------------------------------------------

F_REF = 3.0            # N per pad (force task)
CLOSURE0 = 0.905       # of GRASP: the closing targets that seat every pad on the cylinder
TORQUE0 = 0.5          # the settled targets back off until each servo is at this fraction of its cap
RAMP, HOLD = 200, 200  # settle steps (dt 5 ms): close onto the cylinder, then hold
BACK_RATE = 0.05       # rad/s of the back-off ramps (under the servo's top speed cap/kd = 0.18 rad/s)
BACK_ROUNDS = 6        # relaxing one joint re-seats the others; repeat until no servo is saturated
NUM_STEPS = 60         # rollout length (0.3 s)
RAMP_STEPS = 20        # the targets ramp from the settled ones to theta over the first steps, then hold
LOSS_WINDOW = 10       # the loss averages the last steps (the servo's slow pole has settled)
NEWTON_TOL = 1e-9      # forward Newton tolerance: bounds the gradient fidelity
FORCE_TARGET_ITERS = 70  # the force task run that defines the field task's target grip
FORCE_TARGET_LR = 5e-4
EPS_NORM = 1e-6        # |F|_eps = sqrt(F.F + eps^2): the force-norm loss stays smooth at F = 0


def configure(scene, tol=NEWTON_TOL):
    """A tight forward Newton solve and a tight adjoint solve (as in the other demos)."""
    diffsim.make_scene_differentiable(scene)
    sp = scene.get_solver_params()
    nl = sp.non_linear_solver
    nl.max_iter = 300
    nl.abs_tol = tol
    nl.rel_tol = tol
    sp.non_linear_solver = nl
    scene.set_solver_params(sp)
    bp = diffsim.get_back_propagation_solver_params(scene)
    bp.outer_solver_abs_tol = 1e-10
    bp.outer_solver_max_iter = 300
    bp.inner_solver_abs_tol = 1e-14
    bp.validate_finite_diff = True
    diffsim.set_back_propagation_solver_params(scene, bp)


def effort_caps(rig):
    prefab = rig.bot.get_bot_prefab()
    caps = {prefab.joints[i].name: prefab.joints[i].effort_limit for i in range(len(prefab.joints))}
    return np.array([caps[n] for n in rig.joint_names])


def saturation(rig, theta):
    """The elastic torque demanded by each servo as a fraction of its cap (>= 1: saturated)."""
    return GAIN[0] * np.abs(theta - rig.pose()) / effort_caps(rig)


def pad_forces(rig):
    return np.array([np.linalg.norm(sum(np.asarray(a.get_contact_force_world().tolist()) for a in rig.parts[f]))
                     for f in FINGERS])


class PadForceLoss:
    """0.5 * weight * (|F|_eps - f_ref)^2 on the total world contact force of one pad."""

    def __init__(self, actors, f_ref, weight=1.0):
        self.actors = list(actors)
        self.f_ref = float(f_ref)
        self.weight = float(weight)

    def force(self):
        return sum(np.asarray(a.get_contact_force_world().tolist(), dtype=np.float64) for a in self.actors)

    def value(self):
        n = np.sqrt(self.force() @ self.force() + EPS_NORM ** 2)
        return 0.5 * self.weight * (n - self.f_ref) ** 2

    def accumulate_output_grad(self):
        F = self.force()
        n = np.sqrt(F @ F + EPS_NORM ** 2)
        g = np.ascontiguousarray(self.weight * (n - self.f_ref) / n * F)
        for a in self.actors:  # the pad force is the sum over its parts: the same gradient for each
            diffsim.get_contact_force_world_backward(a, g)


def quat_to_mat_torch(q):
    """xyzw quaternion tensor -> rotation matrix tensor."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)])])


class TaxelField:
    """The 5 x 120 compression map of the hand as a torch function of the engine state.

    For each pad the contacts of its parts (distal link, back cover, tip) against other bodies
    are read from the engine; each contact's taxel weights follow ``project_finger`` and are
    constants of the state, since the sample sits at a fixed point of the (welded) part. The map
    is ``n_j = max(0, -(sum_k w_kj R^T s_k F_k) . taxel_normal_j)`` with F_k the reported world
    force of contact k, s_k its sign and R the link rotation: the torch leaves are the reported
    forces (one (n, 3) tensor per part) and the link quaternions, and their gradients are handed
    to the engine by ``diffsim.get_contact_points_backward`` and ``diffsim.get_root_transform_backward``."""

    def __init__(self, rig):
        self.rig = rig
        self.taxel_nrm = [torch.tensor(rig.layout.normal[fi]) for fi in range(5)]

    def read(self):
        """The engine state as constants and leaves: per finger (q leaf, parts), with
        parts = [(actor, keep, signs, weights (n, 120), F leaf (n, 3))]."""
        rig = self.rig
        fingers = []
        for fi, f in enumerate(FINGERS):
            t = rig.distal[f].get_root_transform()
            q = np.array(t.rotation.tolist())
            o = np.array(t.translation.tolist())
            R = quat_to_mat(q)
            parts = []
            for actor in rig.parts[f]:
                h = actor.get_handle()
                keep, signs, pos, out, force = [], [], [], [], []
                for k, cp in enumerate(actor.get_contact_points_world()):
                    if cp.actor_a == h:
                        other, sign, p, ow = cp.actor_b, 1.0, cp.pos_a, -np.array(cp.normal.tolist())
                    else:
                        other, sign, p, ow = cp.actor_a, -1.0, cp.pos_b, np.array(cp.normal.tolist())
                    if other in rig.hand_handles:
                        continue  # self contact: not part of the tactile field
                    if sign < 0:
                        raise NotImplementedError(
                            f"{actor.get_name()}: a contact of another body's sample against the pad; "
                            "its position slides on the pad and has no adjoint")
                    keep.append(k)
                    signs.append(sign)
                    pos.append(p.tolist())
                    out.append(ow)
                    force.append(np.array(cp.force.tolist()))
                if not keep:
                    continue
                pos, out = np.array(pos), np.array(out)
                p_l = (pos - o) @ R
                out_l = out @ R
                d = np.linalg.norm(p_l[:, None, :] - rig.layout.pos[fi][None], axis=2)
                facing = (out_l @ rig.layout.normal[fi].T) >= COS_MIN
                kern = wendland(d / RADIUS) * facing
                ssum = kern.sum(1)
                w = np.zeros_like(kern)
                covered = ssum > 0.0
                w[covered] = kern[covered] / ssum[covered, None]
                parts.append((actor, np.array(keep), np.array(signs), torch.tensor(w),
                              torch.tensor(np.array(force), requires_grad=True)))
            fingers.append((torch.tensor(q, requires_grad=True), parts))
        return fingers

    def forward(self, fingers):
        """(5, 120) compression tensor from the leaves of read()."""
        rows = []
        for fi, (q, parts) in enumerate(fingers):
            R = quat_to_mat_torch(q)
            t = torch.zeros(120, 3, dtype=torch.float64)
            for actor, keep, signs, w, F in parts:
                f_l = (torch.tensor(signs).reshape(-1, 1) * F) @ R  # forces on the pad, link frame
                t = t + w.T @ f_l
            rows.append(torch.relu(-(t * self.taxel_nrm[fi]).sum(1)))
        return torch.stack(rows)

    def seed(self, fingers):
        """After loss.backward(): hand the leaves' gradients to the engine's adjoints."""
        for fi, (q, parts) in enumerate(fingers):
            if q.grad is not None:
                grad = np.zeros(7)
                grad[3:] = q.grad.numpy()
                diffsim.get_root_transform_backward(self.rig.distal[FINGERS[fi]], np.ascontiguousarray(grad))
            for actor, keep, signs, w, F in parts:
                if F.grad is None:
                    continue
                full = np.zeros((len(actor.get_contact_points_world()), 3))
                full[keep] = F.grad.numpy()
                diffsim.get_contact_points_backward(actor, np.ascontiguousarray(full.reshape(-1)))

    def numpy(self):
        with torch.no_grad():
            return self.forward(self.read()).numpy()


class TaxelMapLoss:
    """0.5 * weight * sum_(pad, taxel) (n - target)^2 on the compression map (target: (5, 120) N)."""

    def __init__(self, field, target, weight=1.0):
        self.field = field
        self.target = torch.tensor(np.asarray(target, dtype=np.float64))
        self.weight = float(weight)

    def loss(self):
        fingers = self.field.read()
        n = self.field.forward(fingers)
        return 0.5 * self.weight * ((n - self.target) ** 2).sum(), fingers

    def value(self):
        with torch.no_grad():
            return float(self.loss()[0])

    def accumulate_output_grad(self):
        # The readout runs inside TorchRollout's autograd.Function forward, where torch has
        # grad mode off: the inner graph (leaves -> map -> loss) needs it back on.
        with torch.enable_grad():
            loss, fingers = self.loss()
            loss.backward()
        self.field.seed(fingers)


def build(closure=CLOSURE0, torque_fraction=TORQUE0):
    """Rig + cylinder in a differentiable scene, settled in an unsaturated grip.

    Closing onto the cylinder stalls every servo short of its target, deep in the torque
    cap where the pad forces no longer depend on the targets. The targets are backed off to
    the reached pose plus ``torque_fraction`` of each cap, along a slow ramp, and the hand
    settles again; relaxing the proximal joints re-seats the distal pads, so the back-off is
    repeated until no servo is saturated. Returns (rig, targets)."""
    rig = Rig(name="xhand tactile diff")
    v, f = make_object(GRASP)
    rig.add_mesh("object", v, f, GRASP["pos"])
    configure(rig.scene)
    for i in range(RAMP):
        rig.set_targets(close_targets(GRASP, closure * i / (RAMP - 1)))
        rig.step()
    rig.step(HOLD)
    for rnd in range(BACK_ROUNDS):
        start = rig.target.copy()
        theta = rig.pose() + torque_fraction * effort_caps(rig) / GAIN[0]
        n = max(int(np.ceil(np.abs(theta - start).max() / (BACK_RATE * DT))), 1)
        for i in range(n):  # a target jump would kick the joint through the damper
            rig.target[:] = start + (theta - start) * (i + 1) / n
            rig.arm.set_articulated_target_pose(np.ascontiguousarray(rig.target))
            rig.step()
        rig.step(HOLD)
        sat = saturation(rig, theta)
        print(f"  back-off round {rnd}: {n} steps, |F| per pad {pad_forces(rig).round(2).tolist()}, "
              f"servo torque / cap max {sat.max():.2f}", flush=True)
        if sat.max() < 1.0:
            return rig, theta.copy()
    raise RuntimeError(f"servos still saturated after {BACK_ROUNDS} back-off rounds: {sat.round(2).tolist()}")


class Demo:
    """The rollout from the settled state: the targets ramp from theta0 (the settled targets) to
    theta over RAMP_STEPS steps and hold; the loss is the tactile error over the last steps."""

    def __init__(self, rig, theta0, losses=None):
        self.rig = rig
        self.theta0 = np.asarray(theta0, dtype=np.float64)
        self.weights = np.minimum(np.arange(1, NUM_STEPS + 1) / RAMP_STEPS, 1.0)
        self.losses = ([PadForceLoss(rig.parts[f], F_REF, 1.0 / LOSS_WINDOW) for f in FINGERS]
                       if losses is None else list(losses))
        self.trace = []  # per rollout: (NUM_STEPS, 5) pad forces
        self.bridge = TorchRollout(rig.scene, dt=DT, num_steps=NUM_STEPS, control_actors=[rig.arm],
                                   step_losses=self.step_losses)

    def step_losses(self, step):
        self.trace[-1][step] = pad_forces(self.rig)
        return self.losses if step >= NUM_STEPS - LOSS_WINDOW else []

    def controls(self, theta):
        """(NUM_STEPS, 12) per-step targets from the (12,) decision variables (torch or numpy)."""
        if isinstance(theta, torch.Tensor):
            w = torch.tensor(self.weights, dtype=torch.float64).reshape(-1, 1)
            return torch.tensor(self.theta0).reshape(1, -1) + w * (theta - torch.tensor(self.theta0)).reshape(1, -1)
        return self.theta0[None, :] + self.weights[:, None] * (np.asarray(theta) - self.theta0)[None, :]

    def loss(self, theta):
        """theta: (12,) torch tensor -> scalar loss tensor (forward + adjoint)."""
        self.trace.append(np.zeros((NUM_STEPS, 5)))
        return self.bridge(controls=self.controls(theta))

    def objective(self, theta_np):
        """The same rollout without the adjoint (for finite differences); leaves the scene at the end."""
        self.trace.append(np.zeros((NUM_STEPS, 5)))
        self.rig.scene.restore_state(self.bridge._state_init, False)
        u = self.controls(theta_np)
        total = 0.0
        for step in range(NUM_STEPS):
            self.rig.arm.set_articulated_target_pose(np.ascontiguousarray(u[step]))
            self.rig.scene.step(DT)
            total += sum(l.value() for l in self.step_losses(step))
        return total


def gradient_check(demo, theta0, eps_list=(1e-5, 1e-6)):
    """Central finite differences of the rollout loss against the adjoint, along the gradient and
    per joint. Contact makes the loss only piecewise smooth, so the two step sizes are printed
    (their disagreement is the finite-difference noise) and the result is reported, not gated."""
    theta = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
    value = demo.loss(theta)
    value.backward()
    g = theta.grad.detach().numpy().copy()
    f0 = demo.objective(theta0)
    print(f"  loss (adjoint run) {float(value.detach()):.9f}  (plain replay) {f0:.9f}")
    d = g / np.linalg.norm(g)
    for eps in eps_list:
        fd = (demo.objective(theta0 + eps * d) - demo.objective(theta0 - eps * d)) / (2 * eps)
        print(f"  along the gradient (eps {eps:.0e}): adjoint {g @ d:+.6e}  fd {fd:+.6e}  "
              f"rel err {abs(g @ d - fd) / abs(fd):.2e}")
    for j, name in enumerate(demo.rig.joint_names):
        e = np.zeros_like(theta0)
        e[j] = eps_list[-1]
        fd = (demo.objective(theta0 + e) - demo.objective(theta0 - e)) / (2 * eps_list[-1])
        print(f"    {name.replace('right_hand_', ''):22s} adjoint {g[j]:+.4e}  fd {fd:+.4e}  "
              f"rel err {abs(g[j] - fd) / max(abs(fd), 1e-30):.1e}")
    return g


def optimize(demo, theta0, iters, lr, decay=1.0, hold=0):
    """Adam on the twelve targets: lr for the first ``hold`` iterations, then a geometric decay
    to lr * decay at the last one."""
    theta = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    hist = []
    for it in range(iters):
        for group in opt.param_groups:
            group["lr"] = lr * decay ** (max(it - hold, 0) / max(iters - 1 - hold, 1))
        opt.zero_grad()
        t0 = time.time()
        value = demo.loss(theta)
        value.backward()
        forces = demo.trace[-1][-LOSS_WINDOW:].mean(0)
        hist.append((float(value.detach()), forces, theta.detach().numpy().copy()))
        r = demo.bridge.last_result
        print(f"  iter {it:3d}  loss {float(value.detach()):.5f}  |F| per pad {forces.round(2).tolist()}  "
              f"|grad| {theta.grad.norm():.3e}  {time.time() - t0:.1f} s  adjoint residual {r.max_adjoint_residual:.1e}",
              flush=True)
        opt.step()
    return hist


# ---------------------------------------------------------------------------
# Rendering (Blender for the hand, a panel for the taxel maps)
# ---------------------------------------------------------------------------

LOOK_FROM = [0.235, -0.150, 0.205]
LOOK_AT = [0.075, -0.005, 0.128]


def render(npz_path, output_dir, samples=96):
    """Replay the saved run and render each recorded iteration: the hand grasping the cylinder
    (Blender) beside the current and target taxel maps. Writes ``xhand_tactile_<task>.mp4`` and
    a gif to ``output_dir``."""
    import imageio.v3 as iio
    from PIL import Image
    from types import SimpleNamespace

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(npz_path, allow_pickle=True)
    thetas, losses, forces = data["theta"], data["loss"], data["forces"]
    task = str(data["task"])
    rig, theta0 = build()
    if not np.allclose(theta0, data["theta0"], atol=1e-9):
        raise RuntimeError("the settled targets differ from the saved run")
    demo = Demo(rig, theta0)
    field = TaxelField(rig) if task == "field" else None
    target = data["target"] if task == "field" else None
    for handle, link in zip(rig.arm.get_nested_link_actors(), rig.prefab_links):
        if link.render_model_file:
            render_model_registry.register(rig.scene, handle, link.render_model_file,
                                           physics.TransformRT(link.render_model_rotation, link.render_model_translation),
                                           link.render_model_scale)
    video.EXPORT_SCENES = True
    export = video.SceneExport(rig.scene, None, LOOK_FROM, LOOK_AT, "", colors={"object": (0.78, 0.36, 0.12)})
    sim_maps = []
    for k, th in enumerate(thetas):
        demo.objective(th)  # leaves the scene at this iteration's rollout end state
        fr, totals = rig.read()
        if field is not None:
            err = np.abs(field.numpy() - fr.normal).max()
            if err > 1e-9:
                raise RuntimeError(f"iteration {k}: torch readout differs from the reader by {err:.1e} N")
        sim_maps.append(fr.normal.copy())
        export.begin_iteration()
        export.capture(None, "", 1, None)
        print(f"  captured iteration {k}: |F| per pad {np.linalg.norm(totals, axis=1).round(2).tolist()}", flush=True)
    stem = output_dir / f"xhand_tactile_{task}"
    export.write(stem)
    frames_dir = output_dir / f"xhand_tactile_{task}_frames"
    blender = os.environ.get("BLENDER", "blender")
    render_script = HERE / "render_diffsim_blender.py"
    subprocess.run([blender, "-b", "--python", str(render_script), "--", "render", str(stem) + ".scene",
                    "--frames-dir", str(frames_dir), "--samples", str(samples), "--size", "960", "540"], check=True)
    scale = 0.5
    target_frame = SimpleNamespace(normal=target) if target is not None else None
    frames = []
    height = 540
    for k in range(len(thetas)):
        blender_frame = np.asarray(Image.open(frames_dir / f"frame_{k:05d}.png").convert("RGB"))
        sim = SimpleNamespace(normal=sim_maps[k])
        if task == "field":
            grids = panel([sim, target_frame], scale, height, width=380, labels=("current", "target"))
        else:
            grids = panel([sim], scale, height, width=200, labels=("current",))
        info = _info_column(k, len(thetas), losses, forces, 260, height, task, target)
        frames.append(np.concatenate([blender_frame, grids, info], axis=1))
    seq = [frames[0]] * 5 + frames + [frames[-1]] * 10
    out = stem.with_suffix(".mp4")
    iio.imwrite(out, np.stack(seq), fps=5, codec="libx264", macro_block_size=1)
    iio.imwrite(stem.with_suffix(".gif"), np.stack(seq), duration=200, loop=0)
    print(f"[render] {len(seq)} frames -> {out} and {stem.with_suffix('.gif')}")


def _info_column(k, n_iter, losses, forces, width, height, task, target):
    """Iteration counter, per-pad force bars against their references and the loss curve so far."""
    from PIL import Image, ImageDraw, ImageFont

    refs = np.full(5, F_REF) if task == "force" else target.sum(1)
    subtitle = "0.5 * sum (|F| - 3 N)^2" if task == "force" else "0.5 * sum_600 (n - n*)^2"
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = small = ImageFont.load_default()
    img = Image.new("RGB", (width, height), (24, 24, 28))
    dr = ImageDraw.Draw(img)
    white, grey, accent = (230, 230, 230), (120, 120, 130), (255, 170, 40)
    dr.text((12, 8), "Differentiable tactile control", fill=white, font=font)
    dr.text((12, 30), f"iteration {k} / {n_iter - 1}", fill=white, font=font)
    dr.text((12, 50), f"loss {losses[k]:.3f}   {subtitle}", fill=grey, font=small)
    dr.text((12, 78), "pad force |F| (N), reference in orange", fill=white, font=small)
    x0, bar_w, f_max = 62, width - 62 - 16, 16.0
    for i, (name, f, ref) in enumerate(zip(FINGERS, forces[k], refs)):
        y = 100 + 26 * i
        dr.text((12, y + 1), name, fill=white, font=small)
        dr.rectangle([x0, y, x0 + bar_w, y + 16], fill=(45, 45, 52))
        dr.rectangle([x0, y, x0 + bar_w * min(f / f_max, 1.0), y + 16],
                     fill=(80, 190, 110) if abs(f - ref) <= 0.5 else (200, 80, 60))
        dr.text((x0 + 4, y + 1), f"{f:.2f}", fill=white, font=small)
        xr = x0 + bar_w * ref / f_max
        dr.line([xr, y - 2, xr, y + 18], fill=accent, width=2)
    top, bottom, left, right = 262, height - 30, 40, width - 12
    dr.text((12, top - 20), "loss (log)", fill=white, font=small)
    dr.rectangle([left, top, right, bottom], outline=(70, 70, 80))
    lo, hi = np.log10(max(min(losses), 1e-3)), np.log10(max(losses))
    pts = []
    for i in range(k + 1):
        x = left + (right - left) * i / max(n_iter - 1, 1)
        y = bottom - (bottom - top) * (np.log10(max(losses[i], 1e-3)) - lo) / max(hi - lo, 1e-9)
        pts.append((x, y))
    if len(pts) > 1:
        dr.line(pts, fill=accent, width=2)
    dr.ellipse([pts[-1][0] - 3, pts[-1][1] - 3, pts[-1][0] + 3, pts[-1][1] + 3], fill=white)
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--task", choices=("force", "field"), default="field")
    ap.add_argument("--check", action="store_true", help="finite-difference gradient check first")
    ap.add_argument("--iters", type=int, default=0)
    ap.add_argument("--lr", type=float, default=8e-4, help="Adam step")
    ap.add_argument("--decay", type=float, default=0.1, help="final / initial learning rate")
    ap.add_argument("--hold", type=int, default=35, help="iterations at the initial rate before the decay")
    ap.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("diffsim_videos"))
    ap.add_argument("--render", type=pathlib.Path, default=None, help="replay a saved run (npz) into the Blender video")
    ap.add_argument("--target", type=pathlib.Path, default=None,
                    help="field task: a saved run whose final targets define the target taxel map "
                         "(default: run the force task first and use its F_REF grip)")
    args = ap.parse_args()

    if args.render is not None:
        render(args.render, args.output_dir)
        return

    lr = args.lr
    t0 = time.time()
    rig, theta0 = build()
    print(f"settled in {time.time() - t0:.1f} s: |F| per pad {pad_forces(rig).round(2).tolist()}  "
          f"servo torque / cap {saturation(rig, theta0).round(2).tolist()}", flush=True)

    target = None
    if args.task == "field":
        field = TaxelField(rig)
        fr, _ = rig.read()
        err = np.abs(field.numpy() - fr.normal).max()
        if err > 1e-9:
            raise RuntimeError(f"torch readout differs from the reader by {err:.1e} N")
        if args.target is not None:
            theta_star = np.load(args.target)["theta"][-1]
            source = args.target.name
        else:
            # The target is the F_REF grip: the force task is run first (from the settled ~12 N
            # grip) and its joint targets define the target map, which the field task then has
            # to recover from the tactile map alone.
            print(f"force task first: regulating every pad to {F_REF:.0f} N for the target grip", flush=True)
            force_demo = Demo(rig, theta0)
            force_hist = optimize(force_demo, theta0, FORCE_TARGET_ITERS, FORCE_TARGET_LR, 1.0, 0)
            theta_star = force_hist[-1][2]
            rig.scene.restore_state(force_demo.bridge._state_init, False)
            force_demo.bridge.close()
            source = f"the force task ({FORCE_TARGET_ITERS} iterations)"
            if args.output_dir is not None:  # reusable as --target for later field runs
                args.output_dir.mkdir(parents=True, exist_ok=True)
                np.savez(args.output_dir / "xhand_tactile_force.npz",
                         loss=np.array([h[0] for h in force_hist]), forces=np.array([h[1] for h in force_hist]),
                         theta=np.array([h[2] for h in force_hist]), theta0=theta0, f_ref=F_REF,
                         joints=np.array(rig.joint_names), task="force")
        probe = Demo(rig, theta0)
        probe.objective(theta_star)
        target = field.numpy()
        rig.scene.restore_state(probe.bridge._state_init, False)
        probe.bridge.close()
        print(f"target map from {source}: {int((target > 0.01).sum())} taxels pressed, "
              f"per pad {target.sum(1).round(2).tolist()} N", flush=True)
        demo = Demo(rig, theta0, [TaxelMapLoss(field, target, 1.0 / LOSS_WINDOW)])
    else:
        demo = Demo(rig, theta0)

    if args.check:
        gradient_check(demo, theta0)
    hist = optimize(demo, theta0, args.iters, lr, args.decay, args.hold) if args.iters else []
    if args.task == "field" and hist:
        err = np.abs(hist[-1][2] - theta_star)
        print(f"recovered the target grip's joint targets to {1e3 * err.max():.2f} mrad "
              f"(started {1e3 * np.abs(theta0 - theta_star).max():.1f} mrad away)", flush=True)
    if args.iters and args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        extra = {} if target is None else {"target": target, "theta_star": theta_star}
        out = args.output_dir / f"xhand_tactile_{args.task}.npz"
        np.savez(out, loss=np.array([h[0] for h in hist]), forces=np.array([h[1] for h in hist]),
                 theta=np.array([h[2] for h in hist]), theta0=theta0, f_ref=F_REF,
                 joints=np.array(demo.rig.joint_names), task=args.task, **extra)
        print(f"wrote {out}")
    demo.bridge.close()
    rig.scene.release_all_states()


if __name__ == "__main__":
    main()
