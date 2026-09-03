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

"""Example: robot-arm trajectory optimization through the differentiable simulator, on video.

A Franka FR3 arm from the SuperDex robotics assets is driven by the engine's
articulated pose controller. The optimization variable is the whole sequence
of per-step joint targets (7 x num_steps); gradients come from the engine's
discrete adjoint through the ``superdex.physics.diffsim_torch`` bridge
(``controls`` group) and are consumed by ``torch.optim.Adam``. Two tasks, each
rendered offscreen with the built-in viewer and written to MP4:

1. ``robot_reach.mp4`` - move the end effector (link ``fr3_link8``) to a
   target point in 1 s from the default posture. No contact.
2. ``robot_push.mp4`` - starting with the end effector already resting against
   a cube on the ground (pre-push pose from the engine's IK solver), sweep the
   cube to a goal that is off the initial push line, so the optimizer has to
   steer it through frictional contact between the arm's collision meshes and
   the cube (articulated-vs-rigid contact in one island; the cube slides on the
   ground with friction).
3. ``robot_grasp.mp4`` - FR3 with the Robotiq 2F-85 gripper descends onto a
   cube, closes the fingers and lifts it straight up; the goal for the cube
   lies 15 cm beside the lift line, so the optimizer has to carry the held
   cube sideways. The variables are knots of the arm's carry-phase targets
   (the fingers keep their closure), updated by normalized gradient descent;
   the loss is the cube's final position, and its gradient reaches the arm
   targets only through the two frictional finger-cube contacts (articulated
   links against a rigid body in one island): the cube moves because the
   fingers hold it.
4. ``finger_tendon.mp4`` - the tendon-driven finger of the physics samples
   (three passive hinges, a prismatic tendon slider, eyelets on the bones)
   actuated by a rod actor as the cable, tied to the slider and the fingertip
   eyelet by node-to-rigid constraints of the cable's own axial stiffness. The
   variables are the per-step slider targets (the tendon pull), updated by
   normalized gradient descent (Adam at a constant rate oscillates around this
   exactly reachable goal); the loss is the fingertip's final position, whose
   goal is the pose a hidden reference pull reaches; the gradient flows from
   the fingertip through the passive hinges, the constraints and the cable's
   elasticity (rod adjoint, with the articulated actor, its controller and the
   rod in one island) into the slider targets.

Rigor. Before optimizing, ``--check`` compares the adjoint gradient of the
largest-gradient target entries against central finite differences of the
full rollout at two step sizes (1e-5, 1e-6) and prints both the relative
error and the FD self-consistency; every forward step is required to reach
``ConvergenceStatus.CONVERGED`` (the script aborts otherwise). What this
shows for the push task (2026-09-01 measurements, frictional arm-cube pair):
the checked entries agree to the FD noise level. Before the two engine fixes
mentioned above the entries at contact onset were off by 5e-2..1.3e-1
although the rollout FD was self-consistent to 1e-5 - friction fading and
the constant stage-start normal, not a smoothness problem.
Design choices that keep the interaction smooth: contact stiffness 1e6, the
end effector starts in contact (no impact), the sweep is slow, and the forward
Newton tolerance is 1e-9 (tighter settings sit at the round-off floor of this
model and stall the solver, which the guard would report).

``robot_push_soft.mp4`` (``--task push_soft``) is the same push with a soft
(neo-Hookean, E = 1e5 Pa) cube. A soft body pressed and dragged by a link can
trap the forward Newton solve in a limit cycle at isolated steps (the
regularized stick stiffness of frictional contact samples competing with the
nodal stiffness; see ContactParams.friction_falloff_vel). Those steps are
solved with failure-adaptive substepping (``PUSH_SUBSTEP_LEVELS`` halvings of
the step, each substep its own adjoint step, see
``superdex.physics.diffsim_rollout``); the guard reports which steps were
split, and the replay used for the video takes the same substeps.
``robot_push_multi.mp4`` (``--task push_multi``) is the push with two cubes in a
row: the end effector pushes the first cube, which pushes the second; the loss
is the second cube's final position, so the gradient crosses two frictional
contacts (arm-cube and cube-cube, both sync contacts of the validated rigid
paths) and the ground friction of both cubes.

``robot_haul.mp4`` (``--task haul``) hauls a box with a cable: a rod actor tied by
node-to-rigid constraints to the FR3's end-effector link and to the top of a box
on the ground; the arm's joint targets are optimized so that the dragged box
ends at a goal off the initial drag line. The gradient flows from the box through
the cable (the rod adjoint) and the constraints into the arm; the only contacts
are the box and the arm against the ground (cable contact is disabled).

``robot_grasp_soft.mp4`` (``--task grasp_soft``) is the grasp with the same soft
cube, solved the same way. Two things differ from the rigid grasp: the fingertip
links get solid-box collision models (the stock fingertip meshes are thin shells
that a soft body's surface samples tunnel through; see
``solid_fingertip_shape_files``, which needs ``h5py``), and the cube's contact
penalty is 1e7 (at 1e6 the contact layer is softer than the material and the
fingers sink through it, see ``GRASP_SOFT_PENALTY``).

Requirements: SUPERDEX_PRECISION=double (set below), the robotics extension
(``superdex.robotics``), polyscope, imageio+ffmpeg, OpenCV, PyTorch, h5py (soft
grasp only), and the repository assets (resolved through
``superdex.physics.paths``). Run::

    python example_diffsim_robot_video.py --output-dir ./diffsim_videos --check
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

os.environ.setdefault("SUPERDEX_PRECISION", "double")

import numpy as np
import superdex.physics as physics
import superdex.robotics as robotics
import torch
from superdex.physics.diffsim_rollout import step_with_substeps
from superdex.physics.diffsim_torch import TorchRollout
from superdex.physics.paths import resolve_asset, resolve_asset_root
from superdex.physics.utils import render_model_registry

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from example_diffsim_video import GRAVITY, Recorder, box_tet_mesh, save_loss_curve  # noqa: E402

diffsim = physics.diffsim

ARM_BOT = "bots/arms/fr3/fr3.superdex_bot"
EE_LINK = "fr3_link8"
GRIPPER_BOT = "bots/arm_hand_combos/fr3_v2_2f_85/fr3_v2_2f_85.superdex_bot"
GRIPPER_EE_LINK = "2f_85_base_link"
GRIPPER_GAINS = (8.0, 0.8)
# The grasp task drives the arm six times stiffer than the reach/push tasks so that the
# gripper (which adds 1 kg at the wrist) settles within a few millimetres of the IK pose
# under gravity before the fingers close, and grasps with a higher friction material.
GRASP_ARM_GAIN_SCALE = 6.0
GRASP_CONTACT = physics.ContactParams(penalty_coefficient=1e6, coulomb_friction_coefficient=0.8)
GRASP_CUBE_HALF = 0.025
GRASP_CLOSE0 = 0.3  # initial finger closure [rad]: holds the cube through a gentle lift
GRASP_CUBE_DENSITY = 2000.0  # [kg/m^3]: a 5 cm cube of 0.25 kg
GRASP_LIFT_STEPS = 45  # initial lift: 25 cm in 0.9 s
GRASP_GOAL_OFFSET = np.array([0.0, 0.15, 0.0])  # the goal lies 15 cm beside the lift line
GRASP_CARRY_START = 65  # first optimized step: the fingers are closed and the cube is 5 cm up
GRASP_NUM_KNOTS = 4  # carry-phase targets = linear interpolation of knots; the first is fixed
NGD_DECAY = 0.95  # per-iteration decay of the normalized-gradient step length
GRASP_TIP_AHEAD = 0.012  # the pose controller lags 1 cm behind the IK pose during the descent
# Per-joint PD gains of the pose controller (stiffness [N m/rad], damping [N m s/rad]).
JOINT_GAINS = list(zip([400, 400, 300, 300, 150, 100, 60], [40, 40, 30, 30, 15, 10, 6]))
# Cable haul: a box (10 cm, 0.3 kg) on the ground, a 3 mm cable (E 2e7 Pa: a soft cable, so the
# pull stretches it visibly) from the end effector to the box top; the arm drags the box +y.
HAUL_BOX_HALF = 0.05
HAUL_BOX_DENSITY = 300.0  # [kg/m^3]
HAUL_BOX_START = np.array([0.50, -0.20, HAUL_BOX_HALF - 0.001])
HAUL_EE_START = np.array([0.50, 0.05, 0.35])  # end-effector waypoints of the initial drag
HAUL_EE_END = np.array([0.50, 0.35, 0.35])
HAUL_GOAL = np.array([0.60, 0.05, HAUL_BOX_HALF])  # 10 cm beside the drag line
HAUL_CABLE_RADIUS = 0.003  # [m]
HAUL_CABLE_YOUNG = 2e7  # [Pa]
HAUL_CABLE_DENSITY = 1000.0  # [kg/m^3]
HAUL_CABLE_ELEMENTS = 16
HAUL_DT, HAUL_STEPS = 0.02, 60
HAUL_NEWTON_TOL = 1e-7  # the cable island's Newton residual stalls at ~5e-9 (round-off)
HAUL_STALL_TOLERANCE = 1e-6
# Under aggressive trajectories a step of the cable island occasionally runs out of Newton
# iterations at ~2e-6 residual (a slack, buckling cable); such steps are substepped.
HAUL_SUBSTEP_LEVELS = 2
HAUL_LEARNING_RATE = 0.004  # Adam; 0.002 halves the loss in 40 iterations, 0.004 reaches 0.00096 in 120
TENDON_SCENE = "samples/tendon_comparison_articulation.mochi_scene"
TENDON_SLIDER_GAINS = (200.0, 5.0)  # pose-controller gains of the tendon slider (prismatic joint)
TENDON_HINGE_DAMPING = 0.02  # the finger hinges are passive: no stiffness, light damping
TENDON_RADIUS = 0.003  # [m]
TENDON_YOUNG = 2e7  # [Pa]: a soft cable, so the pull stretches it visibly
TENDON_DENSITY = 1000.0  # [kg/m^3]
TENDON_ELEMENTS = 16
TENDON_DT, TENDON_STEPS = 0.01, 40
TENDON_PULL0, TENDON_PULL_GOAL = 0.03, 0.09  # [m] slider travel: the initial ramp, the hidden reference
TENDON_NEWTON_TOL = 5e-7  # the finger-tendon island's Newton residual stalls at ~1e-7 (round-off; ~10 N forces)
TENDON_FINGERTIP = np.array([0.1, 0.0, 0.0])  # the distal end of the last bone (a 0.2 m box) in its frame
CUBE_HALF = 0.05
# Soft push: a neo-Hookean cube of the same size. The cube's friction falloff velocity is
# widened so that the stick stiffness of a contact sample, 2 mu N / (falloff dt), stays well
# below the nodal stiffness E h (ContactParams.friction_falloff_vel); the pair value with the
# arm links is the geometric mean of the two actors' values.
PUSH_SOFT_YOUNG = 1e5  # [Pa]
PUSH_SOFT_POISSON = 0.45
PUSH_SOFT_DENSITY = 300.0  # [kg/m^3]
PUSH_SOFT_MASS_DAMPING = 1.0  # [1/s]
PUSH_SOFT_CELLS = 3  # tet-mesh resolution per side
PUSH_SOFT_FALLOFF = 0.1  # [m/s]
PUSH_SUBSTEP_LEVELS = 2  # failure-adaptive substepping: at most dt/4
# The soft cube's Newton solve stalls on round-off at ~1e-5 residual (1e6 penalty samples on a
# 1e5 Pa body; the arm-only tasks stall at ~1e-9): stalls below this are accepted, the limit-cycle
# failures sit at 2e-3..4e-2.
PUSH_SOFT_STALL_TOLERANCE = 1e-4
# Soft grasp: the same cube as the rigid grasp (5 cm, 0.25 kg) as a neo-Hookean body.
GRASP_SOFT_YOUNG = 1e5  # [Pa]
GRASP_SOFT_FALLOFF = 0.1  # [m/s] friction falloff velocity of the cube (see PUSH_SOFT_FALLOFF)
# Contact stiffness of the soft cube's surface samples. At 1e6 the contact layer under a
# fingertip (about 700 N/m over the covered samples) is softer than the cube's material, so the
# fingertips sink through the layer instead of indenting the cube and the grasp slips during
# the lift; at 1e7 the cube is carried (fingertip forces 6-7 N at closure, 1.2-1.7 N in the air).
GRASP_SOFT_PENALTY = 1e7
CONTACT = physics.ContactParams(penalty_coefficient=1e6, coulomb_friction_coefficient=0.4)
# The arm's links carry the same frictional material as the cube: the contact
# pair combines both owners' coefficients by geometric mean, so the arm pushes
# the cube through frictional contact (and the cube slides on the ground with
# friction). Both are differentiated exactly since 2026-09-01: differentiable
# scenes switch friction fading by normal alignment off (see
# ExperimentalEvalParams.fade_friction) and the previous-state adjoint
# differentiates the explicit stage-start contact normal with the SDF Hessian
# (test_diffsim_gradients: test_frictional_contact_through_sdf_edge_regions_is_exact).
ARM_CONTACT = CONTACT
CUBE_CONN = np.array(
    [0, 1, 2, 4, 6, 7, 4, 2, 5, 4, 7, 1, 3, 2, 1, 7, 1, 2, 4, 7], dtype=np.int32
)


def cube_coords(half: float) -> np.ndarray:
    s = half
    return np.array(
        [-s, -s, -s, s, -s, -s, -s, s, -s, s, s, -s, -s, -s, s, s, -s, s, -s, s, s, s, s, s]
    )


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------


def solid_fingertip_shape_files(cache_dir: pathlib.Path) -> dict[str, str]:
    """Solid-box collision models for the two 2F-85 fingertip links, written to
    ``cache_dir`` (``2f_85_<side>_finger_tip_box.mochi.h5``), keyed by link name.

    The stock fingertip collision meshes are thin shells: their SDF is at most 9 mm
    deep and the center of their bounding box lies outside the solid. That is fine
    for the fingers' own contact samples against an object's SDF (the rigid grasp),
    but a soft body's surface samples tunnel through such a shell - the fingertips
    sink into the soft cube with 0.05 N instead of 4 N of contact force. A soft
    body has no SDF of its own, so for the soft grasp the fingertip links carry a
    solid box with the mesh's extents in the link frame instead (the links keep
    their mass properties and render models). Requires ``h5py``."""
    import h5py

    cache_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for side in ("left", "right"):
        source = resolve_asset(f"bots/grippers/2f_85/collision/2f_85_{side}_finger_tip.mochi.h5")
        with h5py.File(source, "r") as model:
            coords = np.asarray(model["mesh/coordinates"], dtype=np.float64)
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        # Corner order of cube_coords(): bit 0 -> x, bit 1 -> y, bit 2 -> z.
        corners = np.array(
            [[hi[0] if i & 1 else lo[0], hi[1] if i & 2 else lo[1], hi[2] if i & 4 else lo[2]] for i in range(8)],
            dtype=np.float32,
        )
        path = cache_dir / f"2f_85_{side}_finger_tip_box.mochi.h5"
        with h5py.File(path, "w") as out:
            mesh = out.create_group("mesh")
            mesh.create_dataset("coordinates", data=corners)
            mesh.create_dataset("connectivity", data=CUBE_CONN.reshape(-1, 4))
        files[f"2f_85_{side}_finger_tip_link"] = str(path)
    return files


def spawn_bot(
    scene,
    bot_path: str,
    ee_link: str,
    gains_of,
    with_controller: bool = True,
    with_contact: bool = True,
    contact=None,
    link_shape_files=None,
):
    """A bot prefab with the given contact material and (optionally) a pose
    controller whose gains come from ``gains_of(joint_name) -> (k, d)`` for each
    revolute joint (welded and closed-loop joints get no tracking term). Returns
    (bot, articulated actor, ``ee_link`` actor, robotics context); the context
    must outlive the bot. ``link_shape_files`` maps link names (suffix match) to
    collision model files that replace the prefab's."""
    prefab = robotics.load_bot_prefab_from_file(str(resolve_asset(bot_path)))
    contact = ARM_CONTACT if contact is None else contact
    replaced = set()
    for i in range(len(prefab.links)):
        link = prefab.links[i]
        link.contact = contact
        if not with_contact:
            link.collider_type = physics.ColliderType.NONE
        for name, path in (link_shape_files or {}).items():
            if link.name.endswith(name):
                link.shape_file = path
                replaced.add(name)
        prefab.links[i] = link
    missing = set(link_shape_files or {}) - replaced
    if missing:
        raise ValueError(f"no link of {bot_path} matches the shape overrides {sorted(missing)}")
    context = robotics.create_context()
    bot = robotics.create_bot(scene, prefab, context)
    arm = bot.get_articulated_actor()
    if with_controller:
        tracking = []
        for i in range(len(prefab.joints)):
            joint = prefab.joints[i]
            if joint.type == physics.ArticulatedJointType.REVOLUTE:
                k, d = gains_of(joint.name)
                tracking.append(
                    physics.PoseTrackingParams(stiffness=float(k), damping=float(d), saturation=-1.0)
                )
            else:  # welded joints get no tracking term
                tracking.append(physics.PoseTrackingParams(stiffness=0.0, damping=0.0, saturation=-1.0))
        arm.add_articulated_pose_controller(physics.PoseControllerParams(joint_tracking=tracking))
    # Visual meshes for the viewer (the same registration superdex_lab performs).
    links = bot.get_bot_prefab().links
    for handle, link in zip(arm.get_nested_link_actors(), links):
        if link.render_model_file:
            render_model_registry.register(
                scene,
                handle,
                link.render_model_file,
                physics.TransformRT(link.render_model_rotation, link.render_model_translation),
                link.render_model_scale,
            )
    actors = []
    scene.for_each_actor(lambda a: actors.append(a) if a.is_nested_link_actor() else None)
    ee = [a for a in actors if a.get_name().endswith(ee_link)][0]
    return bot, arm, ee, context


def _arm_gains():
    gains = iter(JOINT_GAINS)
    return lambda name: next(gains)


def spawn_arm(scene, with_controller: bool = True, with_contact: bool = True):
    """FR3 arm; see :func:`spawn_bot`."""
    return spawn_bot(scene, ARM_BOT, EE_LINK, _arm_gains(), with_controller, with_contact)


def spawn_gripper_arm(
    scene, with_controller: bool = True, with_contact: bool = True, link_shape_files=None
):
    """FR3 arm with the Robotiq 2F-85 gripper (a closed-loop linkage: two of its
    joints are welded cycle closures, six are revolute); the gripper's revolute
    joints get soft gains, the arm's the reach/push gains times
    GRASP_ARM_GAIN_SCALE. The returned end-effector actor is the gripper base.
    ``link_shape_files``: see :func:`spawn_bot`."""
    arm_gains = _arm_gains()

    def gains_of(name):
        if name.startswith("fr3"):
            k, d = arm_gains(name)
            return GRASP_ARM_GAIN_SCALE * k, GRASP_ARM_GAIN_SCALE * d
        return GRIPPER_GAINS

    return spawn_bot(
        scene,
        GRIPPER_BOT,
        GRIPPER_EE_LINK,
        gains_of,
        with_controller,
        with_contact,
        GRASP_CONTACT,
        link_shape_files,
    )


def configure(scene, newton_tol: float = 1e-9) -> None:
    diffsim.make_scene_differentiable(scene)
    solver = scene.get_solver_params()
    newton = solver.non_linear_solver
    newton.max_iter = 300
    # 1e-9 is the tightest tolerance the arm models reach reliably: with the pose
    # controller active the Newton residual stalls at 1e-10..2e-10 (round-off).
    newton.abs_tol = newton_tol
    newton.rel_tol = newton_tol
    solver.non_linear_solver = newton
    scene.set_solver_params(solver)
    params = diffsim.get_back_propagation_solver_params(scene)
    params.outer_solver_abs_tol = 1e-10
    params.outer_solver_max_iter = 300
    params.inner_solver_abs_tol = 1e-14
    params.validate_finite_diff = True
    diffsim.set_back_propagation_solver_params(scene, params)


def ik_joint_poses(waypoints, spawner=None):
    """Joint poses reaching the end-effector waypoints, from a dedicated
    zero-gravity IK scene (the engine solves IK as a quasi-static simulation).
    ``spawner`` builds the bot (default: the FR3 arm)."""
    spawner = spawn_arm if spawner is None else spawner
    scene = physics.create_scene("ik")
    scene.set_gravity([0.0, 0.0, 0.0])
    bot, arm, ee, context = spawner(scene, with_controller=False, with_contact=False)
    solver = physics.experimental.create_ik_solver(scene)
    params = solver.get_solver_params()
    # The IK defaults (abs_tol 1e-2, 1 cm position threshold) treat targets a few
    # centimetres away as already reached; tighten them so every waypoint is solved.
    params.max_iter = 500
    params.abs_tol = 1e-8
    params.rel_tol = 1e-10
    params.position_error_thres = 1e-4
    solver.set_solver_params(params)
    poses = []
    for point in waypoints:
        solver.clear_position_target(ee.get_handle())
        solver.create_position_target(ee.get_handle(), [0.0, 0.0, 0.0], list(point), 1.0)
        solver.solve_ik()
        q = np.zeros(arm.get_num_dofs())
        arm.get_articulated_pose(q)
        reached = np.asarray(ee.get_center_of_mass_transform().translation)
        print(f"  IK waypoint {np.round(point, 3)} -> ee {reached.round(3)}")
        poses.append(q)
    # Tear down in the documented order: targets, the bot, then the solver, which
    # destroys the IK scene it owns (the scene must not be destroyed separately).
    solver.clear_position_target(ee.get_handle())
    robotics.destroy_bot(scene, bot)
    physics.experimental.destroy_ik_solver(solver)
    return poses


def interpolate_targets(keys, num_steps: int) -> np.ndarray:
    """Piecewise-linear joint targets through (step, pose) keys."""
    targets = np.zeros((num_steps, len(keys[0][1])))
    for (s0, qa), (s1, qb) in zip(keys[:-1], keys[1:]):
        for s in range(s0, min(s1, num_steps)):
            a = (s - s0) / (s1 - s0)
            targets[s] = (1.0 - a) * qa + a * qb
    targets[keys[-1][0] :] = keys[-1][1]
    return targets


# ---------------------------------------------------------------------------
# Verification helpers (anti-cheating: FD checks and convergence assertions)
# ---------------------------------------------------------------------------


def check_gradient(
    bridge, controls: np.ndarray, grad: np.ndarray, num_entries: int = 8, max_per_step: int = 2
) -> None:
    """Central finite differences of the full rollout loss on the entries with
    the largest adjoint gradient (at most ``max_per_step`` per step, so the
    check also covers the adjoint solves of earlier steps), at two step sizes
    (1e-5, 1e-6), plus the directional derivative along the gradient itself.
    An entry validates the adjoint only where the two FD estimates agree with
    each other (the "fd self-consistency" column). Single-entry quotients of
    stiff targets (the arm at 6x gains) are dominated by the forward solve's
    convergence noise (about 1e-9 relative in the loss), and larger steps do
    not help - the loss is nonlinear at the 1e-4 rad scale through the finger
    contacts; the directional derivative spreads the perturbation over every
    entry, so its quotient is an order of magnitude cleaner and validates the
    whole gradient at once."""
    loss_of = lambda c: float(bridge(controls=torch.tensor(c, dtype=torch.float64)).detach())
    print("  gradient check (adjoint vs rollout central FD at eps 1e-5 / 1e-6):")
    direction = grad / np.linalg.norm(grad)
    fds = [
        (loss_of(controls + eps * direction) - loss_of(controls - eps * direction)) / (2.0 * eps)
        for eps in (1e-5, 1e-6)
    ]
    print(
        f"    along the gradient: |adjoint| {np.linalg.norm(grad):.5e}  fd {fds[0]:+.5e}  "
        f"rel err {abs(np.linalg.norm(grad) - fds[0]) / abs(fds[0]):.1e}  "
        f"fd self-consistency {abs(fds[0] - fds[1]) / abs(fds[0]):.0e}"
    )
    entries = []
    per_step = {}
    for flat in np.argsort(-np.abs(grad).ravel()):
        s, d = divmod(int(flat), controls.shape[1])
        if grad[s, d] == 0.0 or per_step.get(s, 0) >= max_per_step:
            continue
        per_step[s] = per_step.get(s, 0) + 1
        entries.append((s, d))
        if len(entries) == num_entries:
            break
    rows = []
    for s, d in entries:
        fds = []
        for eps in (1e-5, 1e-6):
            plus, minus = controls.copy(), controls.copy()
            plus[s, d] += eps
            minus[s, d] -= eps
            fds.append((loss_of(plus) - loss_of(minus)) / (2.0 * eps))
        denom = max(abs(fds[0]), 1e-30)
        rows.append((abs(fds[0] - fds[1]) / denom, s, d, fds[0], abs(grad[s, d] - fds[0]) / denom))
    for fd_self, s, d, fd, rel in sorted(rows):
        smooth = "smooth" if fd_self < 1e-4 else "rough "
        print(
            f"    step {s:3d} joint {d:2d}: adjoint {grad[s, d]:+.5e}  fd {fd:+.5e}  "
            f"rel err {rel:.1e}  fd self-consistency {fd_self:.0e}  [{smooth}]"
        )


class ConvergenceGuard:
    """Checks every forward step of a replay: CONVERGED is required, except that a
    solve that stalled (STOPPED) below ``stall_tolerance`` - the round-off floor of
    these robot models, 1e-9 relative to the ~1e2 N force scale - is counted and
    reported rather than treated as a failure. Anything worse aborts the run."""

    stall_tolerance = 1e-7

    def __init__(self, scene, stall_tolerance: float | None = None):
        self.scene = scene
        if stall_tolerance is not None:
            self.stall_tolerance = stall_tolerance
        self.checked = 0
        self.stalled = 0
        self.splits: list[tuple[int, int]] = []  # (step, number of substeps) of split steps

    def note_substeps(self, step: int, taken: list) -> None:
        if len(taken) > 1:
            self.splits.append((step, len(taken)))

    def assert_last_step(self) -> None:
        stats = self.scene.get_solver_stats()
        status = stats.convergence_status
        if status == physics.ConvergenceStatus.CONVERGED:
            pass
        elif (
            status == physics.ConvergenceStatus.STOPPED
            and stats.residual_norm <= self.stall_tolerance
        ):
            self.stalled += 1
        else:
            raise RuntimeError(
                f"forward Newton solve did not converge (status {status}, "
                f"residual {stats.residual_norm:.2e}, iters {stats.max_non_linear_iters})"
            )
        self.checked += 1

    def report(self, name: str) -> None:
        print(
            f"[{name}] forward convergence verified on {self.checked} replayed steps "
            f"({self.stalled} stalled below {self.stall_tolerance:.0e} residual, "
            f"{len(self.splits)} split into substeps: {self.splits[:8]})"
        )


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def run_task(
    name: str,
    scene,
    arm,
    controls0: np.ndarray,
    loss,
    tracked_point,
    target,
    look_from,
    look_at,
    title: str,
    output_dir: pathlib.Path,
    num_iterations: int,
    learning_rate: float,
    dt: float,
    check: bool,
    grad_clip: float = 1.0,
    trainable: np.ndarray | None = None,
    parametrize=None,
    params0: np.ndarray | None = None,
    optimizer_kind: str = "adam",
    curves=None,
    substep_levels: int = 0,
    stall_tolerance: float | None = None,
) -> None:
    """Optimizes the rollout loss over the pose-controller targets.

    By default the variables are the per-step targets themselves (``controls0``),
    updated by Adam with the gradient clipped to ``grad_clip``; ``trainable`` (a
    boolean mask of ``controls0``'s shape) restricts that to a subset of the
    targets. With ``parametrize`` (a torch function mapping a parameter tensor
    to the full targets) and ``params0`` the variables are those parameters
    instead, and ``trainable`` only selects the entries of the target gradient
    that the check validates. ``optimizer_kind`` is ``"adam"`` or ``"ngd"``
    (normalized gradient descent: a step of length ``learning_rate`` along the
    gradient, decaying by NGD_DECAY per iteration - the update then follows the
    gradient's direction instead of moving every variable by the learning rate
    at once, which matters when a jerk can break a contact). ``curves`` is passed
    to the recorder (extra geometry to draw per frame, e.g. a rod's centerline).
    ``substep_levels`` > 0 enables failure-adaptive substepping in the rollout
    and in the replay: a step whose Newton solve fails (not converged and above
    the guard's stall tolerance) is redone as 2, 4, ... substeps. ``stall_tolerance``
    overrides the guard's default residual floor for stalled solves."""
    num_steps = controls0.shape[0]
    guard = ConvergenceGuard(scene, stall_tolerance)
    bridge = TorchRollout(
        scene,
        dt=dt,
        num_steps=num_steps,
        control_actors=[arm],
        terminal_losses=[loss],
        max_substep_levels=substep_levels,
        substep_residual_tolerance=guard.stall_tolerance if substep_levels else None,
    )
    recorder = Recorder(
        scene, target, look_from=look_from, look_at=look_at, title=title, curves=curves
    )
    identity = parametrize is None
    if identity:
        parametrize = lambda p: p
        params0 = controls0
    params = torch.tensor(params0, dtype=torch.float64, requires_grad=True)
    if optimizer_kind == "adam":
        optimizer = torch.optim.Adam([params], lr=learning_rate)
    elif optimizer_kind != "ngd":
        raise ValueError(f"unknown optimizer_kind {optimizer_kind!r}")
    mask = None if trainable is None else torch.tensor(trainable, dtype=torch.float64)
    losses = []
    record = {0, 1, 2, 4, 7, 12, 20, 30, num_iterations - 1}

    def replay(targets: np.ndarray, capture: bool, caption_fn=None) -> None:
        scene.restore_state(bridge._state_init, False)
        if capture:
            recorder.begin_iteration()
        for step in range(num_steps):
            arm.set_articulated_target_pose(np.ascontiguousarray(targets[step]))
            if substep_levels:
                taken = step_with_substeps(
                    scene, dt, substep_levels, guard.stall_tolerance, step=step
                )
                guard.note_substeps(step, taken)
            else:
                scene.step(dt)
            guard.assert_last_step()
            if capture:
                recorder.capture(
                    tracked_point(),
                    caption_fn(step),
                    hold=(1 if step < num_steps - 1 else 12),
                )

    for iteration in range(num_iterations):
        if params.grad is not None:
            params.grad.zero_()
        controls = parametrize(params)
        controls.retain_grad()
        value = bridge(controls=controls)
        value.backward()
        # The gradient check must see the adjoint itself (the gradient w.r.t. the
        # targets), so read it before the clipping below rescales it.
        raw_grad = controls.grad.detach().clone()
        if mask is not None:
            raw_grad *= mask
            if identity:
                params.grad *= mask
        losses.append(float(value.detach()))
        result = bridge.last_result
        if not result.fd_valid:
            print(
                "  warning: the adjoint's finite-difference self-check flagged steps "
                f"{result.flagged_steps}"
            )
        if result.split_steps:
            print(f"  substepped (step, substeps): {result.split_steps}")
        if iteration == 0 and check:
            check_gradient(bridge, controls0, raw_grad.numpy())
        if iteration in record:
            replay(
                controls.detach().numpy().copy(),
                capture=True,
                caption_fn=lambda step, it=iteration: (
                    f"iteration {it}   t = {(step + 1) * dt:.2f} s   loss = {losses[-1]:.5f}"
                ),
            )
        before = params.detach().clone()
        if optimizer_kind == "adam":
            # Contact tasks have occasional nonsmooth steps: cap the update so one
            # spiky gradient cannot throw the arm into an impact (DiffMJX-style clipping).
            torch.nn.utils.clip_grad_norm_([params], max_norm=grad_clip)
            optimizer.step()
        else:
            with torch.no_grad():
                g = params.grad
                params -= (learning_rate * NGD_DECAY**iteration / (g.norm() + 1e-30)) * g
        print(
            f"[{name}] iter {iteration:3d}  loss {losses[-1]:.6f}  "
            f"|grad| {float(raw_grad.norm()):.3e}  update {float((params.detach() - before).norm()):.3e}  "
            f"adjoint residual {result.max_adjoint_residual:.1e}  "
            f"asymmetry {result.max_hessian_asymmetry:.1e}"
        )

    guard.report(name)
    bridge.close()
    recorder.write(output_dir / f"{name}.mp4")
    save_loss_curve(output_dir / f"{name}_loss.png", losses, f"{title.split(':')[0]}: loss vs iteration")


def task_reach(output_dir: pathlib.Path, num_iterations: int, check: bool) -> None:
    scene = physics.create_scene("Differentiable reach")
    scene.set_gravity(GRAVITY)
    bot, arm, ee, context = spawn_arm(scene)
    scene.create_rigid_actor(
        name="ground",
        shape=physics.create_plane_shape(normal=[0, 0, 1], distance=0.0),
        is_static=True,
        contact=CONTACT,
    )
    configure(scene)
    target = np.array([0.55, 0.25, 0.45])
    n = arm.get_num_dofs()
    q0 = np.zeros(n)
    arm.get_articulated_pose(q0)

    class Loss:
        def value(self) -> float:
            d = np.asarray(ee.get_center_of_mass_transform().translation) - target
            return 0.5 * float(d @ d)

        def accumulate_output_grad(self) -> None:
            d = np.asarray(ee.get_center_of_mass_transform().translation) - target
            g = np.zeros(7)
            g[:3] = d
            diffsim.get_center_of_mass_transform_backward(ee, g)

    try:
        run_task(
            "robot_reach",
            scene,
            arm,
            controls0=np.tile(q0, (50, 1)),
            loss=Loss(),
            tracked_point=lambda: np.asarray(ee.get_center_of_mass_transform().translation),
            target=target,
            look_from=[1.7, -1.9, 1.2],
            look_at=[0.4, 0.05, 0.45],
            title="FR3 reach: Adam on the joint-target trajectory (articulated adjoint, torch bridge)",
            output_dir=output_dir,
            num_iterations=num_iterations,
            learning_rate=0.02,
            dt=0.02,
            check=check,
        )
    finally:
        robotics.destroy_bot(scene, bot)
        physics.destroy_scene(scene)


def task_push(output_dir: pathlib.Path, num_iterations: int, check: bool) -> None:
    _task_push_impl(output_dir, num_iterations, check, soft=False)


def task_push_soft(output_dir: pathlib.Path, num_iterations: int, check: bool) -> None:
    _task_push_impl(output_dir, num_iterations, check, soft=True)


def task_push_multi(output_dir: pathlib.Path, num_iterations: int, check: bool) -> None:
    _task_push_impl(output_dir, num_iterations, check, soft=False, multi=True)


PUSH_MULTI_GAP = 0.01  # [m] between the two cubes at the start
PUSH_MULTI_GOAL = np.array([0.93, 0.06, CUBE_HALF])  # for the second cube


def _task_push_impl(
    output_dir: pathlib.Path, num_iterations: int, check: bool, soft: bool, multi: bool = False
) -> None:
    if soft and multi:
        raise ValueError("soft and multi are exclusive")
    name = "robot_push_multi" if multi else ("robot_push_soft" if soft else "robot_push")
    cube_start = np.array([0.55, 0.0, CUBE_HALF - 0.001])
    goal = PUSH_MULTI_GOAL if multi else np.array([0.80, 0.10, CUBE_HALF])
    # Pre-push pose and a straight slow sweep, from IK on the end effector.
    print(f"[{name}] IK for the initial joint-target trajectory")
    # The engine's IK is a quasi-static simulation towards the target, solved
    # from the previous waypoint's pose: sample the sweep densely (5 cm).
    waypoints = [np.array([0.44 + 0.05 * k, 0.0, 0.10]) for k in range(6)]  # 0.44 .. 0.69
    poses = ik_joint_poses(waypoints)
    num_steps = 75
    keys = [(0, poses[0])] + [(15 * k, poses[k]) for k in range(1, 6)]
    controls0 = interpolate_targets(keys, num_steps)

    scene = physics.create_scene("Differentiable push" + (" (soft cube)" if soft else ""))
    scene.set_gravity(GRAVITY)
    bot, arm, ee, context = spawn_arm(scene)
    scene.create_rigid_actor(
        name="ground",
        shape=physics.create_plane_shape(normal=[0, 0, 1], distance=0.0),
        is_static=True,
        contact=CONTACT,
    )
    if soft:
        coordinates, connectivity = box_tet_mesh(size=2.0 * CUBE_HALF, cells=PUSH_SOFT_CELLS)
        rest = coordinates.reshape(-1, 3)
        material = physics.SoftMaterialParams(
            density=PUSH_SOFT_DENSITY, mass_damping_coefficient=PUSH_SOFT_MASS_DAMPING
        )
        material.neo_hookean = physics.NeoHookeanMaterialParams(
            youngs_modulus=PUSH_SOFT_YOUNG, poisson_ratio=PUSH_SOFT_POISSON
        )
        cube = scene.create_soft_actor(
            name="cube",
            shape=physics.create_tet_mesh_shape(coordinates=coordinates, connectivity=connectivity),
            material=material,
            contact=physics.ContactParams(
                penalty_coefficient=CONTACT.penalty_coefficient,
                coulomb_friction_coefficient=CONTACT.coulomb_friction_coefficient,
                friction_falloff_vel=PUSH_SOFT_FALLOFF,
            ),
            world_from_local=physics.TransformRT(cube_start.tolist()),
        )

        def cube_position() -> np.ndarray:
            # Differentiable soft actors keep their root transform fixed; the motion is
            # in the nodal displacements. The tracked point is the mean node position.
            u = np.asarray(cube.get_displacements(), dtype=np.float64).reshape(-1, 3)
            return cube_start + (rest + u).mean(axis=0)

    else:
        cube = scene.create_rigid_actor(
            name="cube",
            shape=physics.create_tet_mesh_shape(coordinates=cube_coords(CUBE_HALF), connectivity=CUBE_CONN),
            density=300.0,
            contact=CONTACT,
            world_from_local=physics.TransformRT(cube_start.tolist()),
        )

        def cube_position() -> np.ndarray:
            return np.asarray(cube.get_center_of_mass_transform().translation)

    if multi:
        # The second cube, in the push line right behind the first; the loss is on it.
        cube2 = scene.create_rigid_actor(
            name="cube2",
            shape=physics.create_tet_mesh_shape(coordinates=cube_coords(CUBE_HALF), connectivity=CUBE_CONN),
            density=300.0,
            contact=CONTACT,
            world_from_local=physics.TransformRT(
                (cube_start + np.array([2.0 * CUBE_HALF + PUSH_MULTI_GAP, 0.0, 0.0])).tolist()
            ),
        )
        target_cube = cube2

        def cube_position() -> np.ndarray:  # noqa: F811 - the tracked object is the second cube
            return np.asarray(cube2.get_center_of_mass_transform().translation)

    else:
        target_cube = cube

    arm.set_articulated_pose_from_joints(poses[0])
    configure(scene)

    class CubeLoss:
        def value(self) -> float:
            d = cube_position() - goal
            return 0.5 * float(d @ d)

        def accumulate_output_grad(self) -> None:
            d = cube_position() - goal
            if soft:
                num_nodes = rest.shape[0]
                diffsim.get_displacements_backward(cube, np.tile(d / num_nodes, num_nodes))
            else:
                g = np.zeros(7)
                g[:3] = d
                diffsim.get_center_of_mass_transform_backward(target_cube, g)

    try:
        run_task(
            name,
            scene,
            arm,
            controls0=controls0,
            loss=CubeLoss(),
            tracked_point=cube_position,
            target=goal,
            look_from=[1.9, -1.6, 0.9],
            look_at=[0.6, 0.05, 0.1],
            title=(
                "FR3 push of a soft cube: Adam on the joint targets, failure-adaptive substeps"
                if soft
                else (
                    "FR3 push of two cubes: Adam on the joint targets through two frictional contacts"
                    if multi
                    else "FR3 push: Adam on the joint-target trajectory through frictional contact"
                )
            ),
            output_dir=output_dir,
            num_iterations=num_iterations,
            learning_rate=0.002,
            dt=0.02,
            check=check,
            grad_clip=0.02,
            substep_levels=PUSH_SUBSTEP_LEVELS if soft else 0,
            stall_tolerance=PUSH_SOFT_STALL_TOLERANCE if soft else None,
        )
    finally:
        robotics.destroy_bot(scene, bot)
        physics.destroy_scene(scene)


def grasp_ik_poses(points):
    """Joint poses of the FR3 + 2F-85 that place the fingertip midpoint (a fixed
    point of the gripper base's local frame while the fingers are open) at each
    point, with the gripper base held at its default, downward orientation."""
    scene = physics.create_scene("ik")
    scene.set_gravity([0.0, 0.0, 0.0])
    bot, arm, base, context = spawn_gripper_arm(scene, with_controller=False, with_contact=False)
    links = []
    scene.for_each_actor(lambda a: links.append(a) if a.is_nested_link_actor() else None)
    tips = [a for a in links if "finger_tip_link" in a.get_name()]
    base_tfm = base.get_root_transform()  # IK local positions are in the link's root frame
    tip_mid = np.mean([np.asarray(t.get_center_of_mass_transform().translation) for t in tips], axis=0)
    local_mid = np.asarray((base_tfm.inverse() * physics.TransformRT(tip_mid.tolist())).translation)
    base_rotation = np.asarray(base_tfm.rotation.to_rotation_vector())
    solver = physics.experimental.create_ik_solver(scene)
    params = solver.get_solver_params()
    params.max_iter = 500
    params.abs_tol = 1e-8
    params.rel_tol = 1e-10
    params.position_error_thres = 1e-4
    solver.set_solver_params(params)
    solver.create_rotation_target(base.get_handle(), [0.0, 0.0, 0.0], base_rotation.tolist(), 1.0)
    poses = []
    for point in points:
        solver.clear_position_target(base.get_handle())
        solver.create_position_target(base.get_handle(), local_mid.tolist(), list(point), 100.0)
        solver.solve_ik()
        q = np.zeros(arm.get_num_dofs())
        arm.get_articulated_pose(q)
        reached = np.mean([np.asarray(t.get_center_of_mass_transform().translation) for t in tips], axis=0)
        print(f"  IK fingertip midpoint {np.round(point, 3)} -> {reached.round(3)}")
        poses.append(q)
    # Tear down in the documented order: targets, the bot, then the solver, which
    # destroys the IK scene it owns (the scene must not be destroyed separately).
    solver.clear_position_target(base.get_handle())
    solver.clear_rotation_target(base.get_handle())
    robotics.destroy_bot(scene, bot)
    physics.experimental.destroy_ik_solver(solver)
    return poses


def build_grasp_task(soft: bool = False, fingertip_box_dir: pathlib.Path | None = None):
    """Scene, actors and initial controls of the grasp task (see :func:`task_grasp`).
    Returns (scene, bot, arm, cube, cube_position, context, controls0, goal, dt);
    ``cube_position()`` is the cube's center of mass (rigid) or mean node position
    (soft, ``soft=True``: a neo-Hookean cube of the same size and mass, gripped by
    solid-box fingertips written to ``fingertip_box_dir``, see
    :func:`solid_fingertip_shape_files`)."""
    if soft and fingertip_box_dir is None:
        raise ValueError("the soft grasp needs fingertip_box_dir for the fingertip collision models")
    cube_pos = np.array([0.50, 0.0, GRASP_CUBE_HALF - 0.001])
    print(f"[{'robot_grasp_soft' if soft else 'robot_grasp'}] IK for the descend / grasp / lift poses")
    grasp_mid = cube_pos + np.array([GRASP_TIP_AHEAD, 0.0, 0.015])
    pre_mid = grasp_mid + np.array([0.0, 0.0, 0.12])
    lift_mid = np.array([cube_pos[0], cube_pos[1], 0.30])
    q_pre, q_grasp, q_lift = grasp_ik_poses([pre_mid, grasp_mid, lift_mid])

    scene = physics.create_scene("robot_grasp_soft" if soft else "robot_grasp")
    scene.set_gravity([0.0, 0.0, -9.81])
    bot, arm, base, context = spawn_gripper_arm(
        scene, link_shape_files=solid_fingertip_shape_files(fingertip_box_dir) if soft else None
    )
    scene.create_rigid_actor(
        name="ground",
        shape=physics.create_plane_shape(normal=[0.0, 0.0, 1.0], distance=0.0),
        is_static=True,
        contact=GRASP_CONTACT,
    )
    if soft:
        coordinates, connectivity = box_tet_mesh(size=2.0 * GRASP_CUBE_HALF, cells=PUSH_SOFT_CELLS)
        rest = coordinates.reshape(-1, 3)
        material = physics.SoftMaterialParams(
            density=GRASP_CUBE_DENSITY, mass_damping_coefficient=PUSH_SOFT_MASS_DAMPING
        )
        material.neo_hookean = physics.NeoHookeanMaterialParams(
            youngs_modulus=GRASP_SOFT_YOUNG, poisson_ratio=PUSH_SOFT_POISSON
        )
        cube = scene.create_soft_actor(
            name="cube",
            shape=physics.create_tet_mesh_shape(coordinates=coordinates, connectivity=connectivity),
            material=material,
            contact=physics.ContactParams(
                penalty_coefficient=GRASP_SOFT_PENALTY,
                coulomb_friction_coefficient=GRASP_CONTACT.coulomb_friction_coefficient,
                friction_falloff_vel=GRASP_SOFT_FALLOFF,
            ),
            world_from_local=physics.TransformRT(cube_pos.tolist()),
        )

        def cube_position() -> np.ndarray:
            u = np.asarray(cube.get_displacements(), dtype=np.float64).reshape(-1, 3)
            return cube_pos + (rest + u).mean(axis=0)

    else:
        cube = scene.create_rigid_actor(
            name="cube",
            shape=physics.create_tet_mesh_shape(
                coordinates=cube_coords(GRASP_CUBE_HALF), connectivity=CUBE_CONN
            ),
            density=GRASP_CUBE_DENSITY,
            contact=GRASP_CONTACT,
            world_from_local=physics.TransformRT(cube_pos.tolist()),
        )

        def cube_position() -> np.ndarray:
            return np.asarray(cube.get_center_of_mass_transform().translation)

    arm.set_articulated_pose_from_joints(q_pre)
    configure(scene)

    num_steps = 100
    dt = 0.02
    n = arm.get_num_dofs()
    gripper = list(range(7, n))
    close_dir = np.sign(q_pre[gripper])  # each finger joint closes away from zero
    controls0 = interpolate_targets(
        [(0, q_pre), (30, q_grasp), (55, q_grasp), (55 + GRASP_LIFT_STEPS, q_lift)], num_steps
    )
    for step in range(num_steps):
        close = 0.0 if step < 30 else min(1.0, (step - 30) / 25.0)
        controls0[step, gripper] = q_pre[gripper] + close_dir * GRASP_CLOSE0 * close
    # Where the cube ends up when it stays in the grasp (under the lifted fingertips),
    # displaced sideways: the optimizer has to carry it there.
    goal = np.array([cube_pos[0] - 0.005, cube_pos[1], 0.27]) + GRASP_GOAL_OFFSET
    return scene, bot, arm, cube, cube_position, context, controls0, goal, dt


def task_grasp(output_dir: pathlib.Path, num_iterations: int, check: bool) -> None:
    _task_grasp_impl(output_dir, num_iterations, check, soft=False)


def task_grasp_soft(output_dir: pathlib.Path, num_iterations: int, check: bool) -> None:
    """The grasp with a soft cube (see :func:`task_grasp`); failure-adaptive substeps as in
    the soft push."""
    _task_grasp_impl(output_dir, num_iterations, check, soft=True)


def _task_grasp_impl(output_dir: pathlib.Path, num_iterations: int, check: bool, soft: bool) -> None:
    """FR3 + 2F-85: descend onto a cube, close the fingers, lift it straight up.
    The goal for the cube lies 15 cm beside the lift line, so the loss (the
    cube's final position) can only be reduced by carrying the held cube
    sideways: its gradient flows from the cube through the two frictional
    finger contacts into the arm's carry-phase targets, the only optimized
    controls. (A dropped cube is a threshold event - the slip of a Coulomb
    contact - after which the final position no longer depends on the
    controls. So the demo starts from a holding grasp, keeps the finger
    closure fixed - with the fingers free, shoving the cube with one finger is
    the steepest descent direction and drops it within a few iterations - and
    parametrizes the carry by knots updated along the gradient's direction:
    Adam's first step moves every per-step target by the learning rate at
    once, and that jerk drops the cube too.)"""
    name = "robot_grasp_soft" if soft else "robot_grasp"
    scene, bot, arm, cube, cube_position, context, controls0, goal, dt = build_grasp_task(
        soft, fingertip_box_dir=output_dir / "fingertip_boxes" if soft else None
    )
    # Optimize the arm's targets of the carry phase; the fingers keep their closure
    # (a finger shoving the cube sideways is the quickest way to move it, and to drop it).
    trainable = np.zeros(controls0.shape, dtype=bool)
    trainable[GRASP_CARRY_START:, :7] = True
    # The carry-phase arm targets are the linear interpolation of GRASP_NUM_KNOTS knots.
    # The first knot is pinned to the trajectory at the carry start, so an update is a
    # ramp starting from rest rather than a jump (a jerk at the carry start drops the
    # cube, as does moving every per-step target by the learning rate at once); the
    # other knots are the optimization variables. The initial carry phase is linear, so
    # the knots reproduce it exactly.
    num_steps = controls0.shape[0]
    knots = np.linspace(GRASP_CARRY_START, num_steps - 1, GRASP_NUM_KNOTS).round().astype(int)
    steps = np.arange(GRASP_CARRY_START, num_steps)
    weights = np.zeros((len(steps), len(knots)))
    for i, step in enumerate(steps):
        k = min(int(np.searchsorted(knots, step, side="right")) - 1, len(knots) - 2)
        t = (step - knots[k]) / (knots[k + 1] - knots[k])
        weights[i, k], weights[i, k + 1] = 1.0 - t, t
    interp = torch.tensor(weights, dtype=torch.float64)
    base = torch.tensor(controls0, dtype=torch.float64)
    fixed_knot = base[knots[0], :7].reshape(1, 7)
    params0 = controls0[knots[1:], :7]

    def parametrize(params):
        carry = interp @ torch.cat([fixed_knot, params], dim=0)
        arm_targets = torch.cat([base[:GRASP_CARRY_START, :7], carry], dim=0)
        return torch.cat([arm_targets, base[:, 7:]], dim=1)

    reproduced = parametrize(torch.tensor(params0, dtype=torch.float64)).numpy()
    assert np.allclose(reproduced, controls0, atol=1e-12), "knots must reproduce the initial carry"

    class CubeLoss:
        def value(self) -> float:
            d = cube_position() - goal
            return 0.5 * float(d @ d)

        def accumulate_output_grad(self) -> None:
            d = cube_position() - goal
            if soft:
                num_nodes = np.asarray(cube.get_displacements()).size // 3
                diffsim.get_displacements_backward(cube, np.tile(d / num_nodes, num_nodes))
            else:
                g = np.zeros(7)
                g[:3] = d
                diffsim.get_center_of_mass_transform_backward(target_cube, g)

    try:
        run_task(
            name,
            scene,
            arm,
            controls0=controls0,
            loss=CubeLoss(),
            tracked_point=cube_position,
            target=goal,
            look_from=[1.4, -1.2, 0.7],
            look_at=[0.5, 0.0, 0.15],
            title=(
                "FR3 + 2F-85 grasp of a soft cube: normalized gradient descent on the carry knots"
                if soft
                else "FR3 + 2F-85 grasp: normalized gradient descent on the carry knots through frictional contact"
            ),
            output_dir=output_dir,
            num_iterations=num_iterations,
            learning_rate=0.02,
            dt=dt,
            check=check,
            trainable=trainable,
            parametrize=parametrize,
            params0=params0,
            optimizer_kind="ngd",
            substep_levels=PUSH_SUBSTEP_LEVELS if soft else 0,
            stall_tolerance=PUSH_SOFT_STALL_TOLERANCE if soft else None,
        )
    finally:
        robotics.destroy_bot(scene, bot)
        physics.destroy_scene(scene)
        del context


def build_cable(scene, start: np.ndarray, end: np.ndarray, name: str):
    """A straight rod actor (the cable) from ``start`` to ``end`` with the haul
    cable's material; returns (rod, rest node positions, edges, node-constraint
    stiffness = EA / L)."""
    ex = physics.experimental
    nodes = start + np.linspace(0.0, 1.0, HAUL_CABLE_ELEMENTS + 1)[:, None] * (end - start)
    tangent = (end - start) / np.linalg.norm(end - start)
    axis = np.array([1.0, 0.0, 0.0])
    axis -= (axis @ tangent) * tangent
    axis /= np.linalg.norm(axis)
    model = ex.generate_tubular_rod_model_data(
        nodes=nodes.tolist(),
        element_frame_axes=[axis.tolist()] * HAUL_CABLE_ELEMENTS,
        radius=HAUL_CABLE_RADIUS,
        num_cross_section_segments=6,
        is_closed_loop=False,
    )
    area = np.pi * HAUL_CABLE_RADIUS**2
    inertia = 0.25 * np.pi * HAUL_CABLE_RADIUS**4
    material = ex.RodMaterialParams(
        linear_density=HAUL_CABLE_DENSITY * area,
        linear_rotational_inertia=HAUL_CABLE_DENSITY * 2.0 * inertia,
        axial_stiffness=HAUL_CABLE_YOUNG * area,
        torsional_stiffness=0.4 * HAUL_CABLE_YOUNG * 2.0 * inertia,  # G = E / (2 (1 + nu)), nu = 0.25
        flexural_stiffness=[HAUL_CABLE_YOUNG * inertia, HAUL_CABLE_YOUNG * inertia],
    )
    rod = ex.create_rod_actor(
        scene,
        ex.RodActorParams(
            name=name, shape=physics.create_model_shape(model), material=material, has_gravity=True
        ),
    )
    edges = np.array([[i, i + 1] for i in range(HAUL_CABLE_ELEMENTS)], dtype=np.int32)
    return rod, nodes, edges, HAUL_CABLE_YOUNG * area / np.linalg.norm(end - start)


def task_haul(output_dir: pathlib.Path, num_iterations: int, check: bool) -> None:
    """FR3 hauling a box with a cable (see the module docstring). The initial
    trajectory drags the box straight along +y; the goal lies 10 cm beside that
    line, so the optimizer has to swing the drag sideways. Rod contact is disabled
    against the arm and the box (the cable is tied to both), the box and the arm
    slide on the ground. Steps whose Newton solve runs out of iterations above
    the stall tolerance are substepped (HAUL_SUBSTEP_LEVELS)."""
    print("[robot_haul] IK for the initial end-effector drag")
    waypoints = [HAUL_EE_START + (HAUL_EE_END - HAUL_EE_START) * k / 5 for k in range(6)]
    poses = ik_joint_poses(waypoints)
    keys = [(0, poses[0])] + [(12 * k, poses[k]) for k in range(1, 6)]
    controls0 = interpolate_targets(keys, HAUL_STEPS)

    scene = physics.create_scene("Differentiable haul")
    scene.set_gravity(GRAVITY)
    bot, arm, ee, context = spawn_arm(scene)
    scene.create_rigid_actor(
        name="ground",
        shape=physics.create_plane_shape(normal=[0, 0, 1], distance=0.0),
        is_static=True,
        contact=CONTACT,
    )
    box = scene.create_rigid_actor(
        name="box",
        shape=physics.create_tet_mesh_shape(coordinates=cube_coords(HAUL_BOX_HALF), connectivity=CUBE_CONN),
        density=HAUL_BOX_DENSITY,
        contact=CONTACT,
        world_from_local=physics.TransformRT(HAUL_BOX_START.tolist()),
    )
    arm.set_articulated_pose_from_joints(poses[0])
    # The cable runs from the end effector's center of mass to the box top.
    start = np.asarray(ee.get_center_of_mass_transform().translation)
    end = HAUL_BOX_START + np.array([0.0, 0.0, HAUL_BOX_HALF])
    rod, rod_reference, rod_edges, stiffness = build_cable(scene, start, end, "cable")
    scene.enable_actor_contact_symmetric(
        rod.get_handle(), arm.get_handle(), False, physics.IncludeNestedActors.YES
    )
    scene.enable_actor_contact_symmetric(
        rod.get_handle(), box.get_handle(), False, physics.IncludeNestedActors.NO
    )
    for node, actor, local in ((0, ee, [0.0, 0.0, 0.0]), (HAUL_CABLE_ELEMENTS, box, [0.0, 0.0, HAUL_BOX_HALF])):
        scene.create_deformable_node_to_rigid_constraint(
            deformable_actor=rod.get_handle(),
            rigid_actor=actor.get_handle(),
            deformable_node_index=node,
            rigid_local_pos=local,
            stiffness=stiffness,
        )
    configure(scene, newton_tol=HAUL_NEWTON_TOL)

    def cable_centerline():
        nodes = rod_reference + np.asarray(rod.get_displacements()).reshape(-1, 4)[:, :3]
        return [("cable", nodes, rod_edges, 0.006, [0.85, 0.65, 0.1])]

    class BoxLoss:
        def value(self) -> float:
            d = np.asarray(box.get_center_of_mass_transform().translation) - HAUL_GOAL
            return 0.5 * float(d @ d)

        def accumulate_output_grad(self) -> None:
            d = np.asarray(box.get_center_of_mass_transform().translation) - HAUL_GOAL
            g = np.zeros(7)
            g[:3] = d
            diffsim.get_center_of_mass_transform_backward(box, g)

    try:
        run_task(
            "robot_haul",
            scene,
            arm,
            controls0=controls0,
            loss=BoxLoss(),
            tracked_point=lambda: np.asarray(box.get_center_of_mass_transform().translation),
            target=HAUL_GOAL,
            look_from=[1.9, -1.4, 0.9],
            look_at=[0.55, 0.05, 0.15],
            title="FR3 cable haul: Adam on the joint targets through the cable and ground friction",
            output_dir=output_dir,
            num_iterations=num_iterations,
            learning_rate=HAUL_LEARNING_RATE,
            dt=HAUL_DT,
            check=check,
            grad_clip=0.02,
            curves=cable_centerline,
            substep_levels=HAUL_SUBSTEP_LEVELS,
            stall_tolerance=HAUL_STALL_TOLERANCE,
        )
    finally:
        robotics.destroy_bot(scene, bot)
        physics.destroy_scene(scene)
        del context


def _rotate(q: np.ndarray, o: np.ndarray) -> np.ndarray:
    """R(q) o for a unit quaternion q = (x, y, z, w)."""
    v, w = q[:3], q[3]
    return o + 2.0 * w * np.cross(v, o) + 2.0 * np.cross(v, np.cross(v, o))


def _rotate_jacobian(q: np.ndarray, o: np.ndarray) -> np.ndarray:
    """d(R(q) o)/dq (3 x 4, columns x, y, z, w) of :func:`_rotate`."""
    v, w = q[:3], q[3]
    jac = np.zeros((3, 4))
    for i in range(3):
        e = np.zeros(3)
        e[i] = 1.0
        jac[:, i] = 2.0 * w * np.cross(e, o) + 2.0 * (
            np.cross(e, np.cross(v, o)) + np.cross(v, np.cross(e, o))
        )
    jac[:, 3] = 2.0 * np.cross(v, o)
    return jac


def link_point(actor, local_offset: np.ndarray):
    """World position of a point fixed in a link's center-of-mass frame, plus the
    quaternion (XYZW) it was computed with."""
    transform = actor.get_center_of_mass_transform()
    q = np.array([transform.rotation[i] for i in range(4)], dtype=np.float64)
    return np.asarray(transform.translation) + _rotate(q, local_offset), q


def build_tendon_finger():
    """The tendon-driven finger of the physics samples (three passive hinges, a
    prismatic tendon slider, eyelets on the bones) with a rod actor as the tendon,
    tied to the slider and the fingertip eyelet by node-to-rigid constraints of the
    cable's own axial stiffness EA/L. The finger root is welded to the world
    (RootFree -> HARD) and a pose controller drives the slider only (the hinges get
    damping, no stiffness). Cable-vs-finger contact is disabled: rod contact adjoints
    exist for static colliders only, and the cable runs through the eyelets anyway.
    Returns (scene, finger actor, rod, fingertip link actor, slider DoF index)."""
    ex = physics.experimental
    prefab = physics.prefab.load_from_file(
        prefab_path=str(resolve_asset(TENDON_SCENE)),
        root_path=str(resolve_asset_root(TENDON_SCENE)),
    )
    finger = prefab.actors.articulated[0]
    finger.name = "finger"
    joint_type = physics.ArticulatedJointType
    tracking = []  # one entry per joint (the controller accepts 1 or num-joints entries)
    for i in range(len(finger.joints)):
        joint = finger.joints[i]
        if joint.name == "RootFree":
            joint.type = joint_type.HARD
            finger.joints[i] = joint
        if joint.type == joint_type.PRISMATIC:
            k, d = TENDON_SLIDER_GAINS
            tracking.append(physics.PoseTrackingParams(stiffness=k, damping=d, saturation=-1.0))
        elif joint.type == joint_type.REVOLUTE:
            tracking.append(
                physics.PoseTrackingParams(
                    stiffness=0.0, damping=TENDON_HINGE_DAMPING, saturation=-1.0
                )
            )
        else:
            tracking.append(physics.PoseTrackingParams(stiffness=0.0, damping=0.0, saturation=-1.0))
    prefab.controllers.append(
        physics.prefab.PoseControllerPrefab(articulated_actor="finger", joint_tracking=tracking)
    )
    scene = physics.create_scene("Differentiable tendon finger")
    scene.set_gravity(GRAVITY)
    added = physics.prefab.add_to_scene(
        prefab=prefab,
        scene=scene,
        params=physics.prefab.PrefabParams(
            name="demo", translation=[0.0, 0.0, 0.5], apply_scene_settings=False
        ),
    )
    actor = added.filter(physics.ActorType.ARTICULATED)[0]
    info = actor.get_articulated_shape_info()
    slider_dof = [
        info.dof_info[i].offset
        for i, name in enumerate(info.joint_names)
        if name == "SliderPrismatic"
    ][0]
    links = []
    scene.for_each_actor(lambda a: links.append(a) if a.is_nested_link_actor() else None)

    def link(suffix):
        return [a for a in links if a.get_name().endswith(suffix)][0]

    slider, eyelet, tip = link("Slider"), link("Eyelet3"), link("Bone3")
    # The fingertip point (link_point) must agree with the engine's transform product.
    engine_tip = tip.get_center_of_mass_transform() * physics.TransformRT(TENDON_FINGERTIP.tolist())
    assert np.allclose(link_point(tip, TENDON_FINGERTIP)[0], np.asarray(engine_tip.translation), atol=1e-12)
    # The cable: a straight rod from the slider to the last eyelet.
    start = np.asarray(slider.get_center_of_mass_transform().translation)
    end = np.asarray(eyelet.get_center_of_mass_transform().translation)
    nodes = start + np.linspace(0.0, 1.0, TENDON_ELEMENTS + 1)[:, None] * (end - start)
    tangent = (end - start) / np.linalg.norm(end - start)
    axis = np.array([0.0, 0.0, 1.0])
    axis -= (axis @ tangent) * tangent
    axis /= np.linalg.norm(axis)
    model = ex.generate_tubular_rod_model_data(
        nodes=nodes.tolist(),
        element_frame_axes=[axis.tolist()] * TENDON_ELEMENTS,
        radius=TENDON_RADIUS,
        num_cross_section_segments=6,
        is_closed_loop=False,
    )
    area = np.pi * TENDON_RADIUS**2
    inertia = 0.25 * np.pi * TENDON_RADIUS**4  # second moment of area of the cross section
    material = ex.RodMaterialParams(
        linear_density=TENDON_DENSITY * area,
        linear_rotational_inertia=TENDON_DENSITY * 2.0 * inertia,
        axial_stiffness=TENDON_YOUNG * area,
        torsional_stiffness=0.4 * TENDON_YOUNG * 2.0 * inertia,  # G = E / (2 (1 + nu)), nu = 0.25
        flexural_stiffness=[TENDON_YOUNG * inertia, TENDON_YOUNG * inertia],
    )
    rod = ex.create_rod_actor(
        scene,
        ex.RodActorParams(
            name="tendon",
            shape=physics.create_model_shape(model),
            material=material,
            layer="Tendon",
            has_gravity=True,
        ),
    )
    for other in ("Bone", "RoutingGuide"):
        scene.enable_layer_contact_symmetric("Tendon", other, False)
    stiffness = TENDON_YOUNG * area / np.linalg.norm(end - start)
    for node, link_actor in ((0, slider), (TENDON_ELEMENTS, eyelet)):
        scene.create_deformable_node_to_rigid_constraint(
            deformable_actor=rod.get_handle(),
            rigid_actor=link_actor.get_handle(),
            deformable_node_index=node,
            rigid_local_pos=[0.0, 0.0, 0.0],
            stiffness=stiffness,
        )
    return scene, actor, rod, tip, slider_dof


def task_tendon(output_dir: pathlib.Path, num_iterations: int, check: bool) -> None:
    """Tendon-driven finger: find the tendon pull (per-step slider targets) that
    brings the fingertip to the pose a hidden reference pull reaches. The
    gradient of the fingertip's final position w.r.t. the slider targets flows
    through the passive hinges, the two node-to-link constraints and the cable's
    elasticity (the rod adjoint); the finger, its controller and the cable form
    one island. Normalized gradient descent with the decaying step reaches the
    goal to round-off in ~15 iterations (Adam at a constant 2e-3 oscillated
    between 1e-7 and 8e-5 in loss, 2026-09-02)."""
    scene, finger, rod, tip, slider_dof = build_tendon_finger()
    try:
        configure(scene, newton_tol=TENDON_NEWTON_TOL)
        n = finger.get_num_dofs()
        q0 = np.zeros(n)
        finger.get_articulated_pose(q0)

        def ramp(pull: float) -> np.ndarray:
            controls = np.tile(q0, (TENDON_STEPS, 1))
            controls[:, slider_dof] = q0[slider_dof] + np.linspace(0.0, pull, TENDON_STEPS)
            return controls

        fingertip = lambda: link_point(tip, TENDON_FINGERTIP)[0]
        # The goal: where the fingertip ends up under the hidden reference pull.
        state0 = scene.capture_state()
        guard = ConvergenceGuard(scene)
        for targets in ramp(TENDON_PULL_GOAL):
            finger.set_articulated_target_pose(np.ascontiguousarray(targets))
            scene.step(TENDON_DT)
            guard.assert_last_step()
        goal = fingertip().copy()
        scene.restore_state(state0, False)
        scene.release_all_states()
        controls0 = ramp(TENDON_PULL0)
        trainable = np.zeros(controls0.shape, dtype=bool)
        trainable[:, slider_dof] = True  # the hinge targets have no stiffness: zero gradient
        rod_reference = np.asarray(rod.get_mesh().coordinates, dtype=np.float64).reshape(-1, 3)
        rod_edges = np.stack(
            [np.arange(TENDON_ELEMENTS), np.arange(1, TENDON_ELEMENTS + 1)], axis=1
        )

        def rod_centerline():
            nodes = rod_reference + np.asarray(rod.get_displacements()).reshape(-1, 4)[:, :3]
            return [("tendon", nodes, rod_edges, 0.008, [0.85, 0.65, 0.1])]

        class TipLoss:
            """0.5 |fingertip - goal|^2; the fingertip is a point fixed in the last
            bone, so the gradient reaches both the translation and the quaternion
            of its center-of-mass transform (the engine converts the latter to its
            Lie-parameterized state gradient)."""

            def value(self) -> float:
                d = fingertip() - goal
                return 0.5 * float(d @ d)

            def accumulate_output_grad(self) -> None:
                point, q = link_point(tip, TENDON_FINGERTIP)
                d = point - goal
                g = np.zeros(7)
                g[:3] = d
                g[3:] = _rotate_jacobian(q, TENDON_FINGERTIP).T @ d
                diffsim.get_center_of_mass_transform_backward(tip, g)

        run_task(
            "finger_tendon",
            scene,
            finger,
            controls0=controls0,
            loss=TipLoss(),
            tracked_point=fingertip,
            target=goal,
            look_from=[0.25, 1.05, 1.0],  # the eyelet side, so the cable and the goal stay visible
            look_at=[0.4, 0.05, 0.5],
            title="Tendon finger: normalized gradient descent on the tendon pull (rod adjoint)",
            output_dir=output_dir,
            num_iterations=num_iterations,
            learning_rate=0.012,
            dt=TENDON_DT,
            check=check,
            trainable=trainable,
            optimizer_kind="ngd",
            curves=rod_centerline,
        )
    finally:
        physics.destroy_scene(scene)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("diffsim_videos"))
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument(
        "--task",
        choices=[
            "reach", "push", "push_soft", "push_multi", "grasp", "grasp_soft", "tendon", "haul", "both", "all"
        ],
        default="all",
    )
    parser.add_argument("--check", action="store_true", help="finite-difference gradient check first")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    physics.initialize(num_worker_threads=0)
    start = time.time()
    if args.task in ("reach", "both", "all"):
        task_reach(args.output_dir, args.iterations, args.check)
    if args.task in ("push", "both", "all"):
        task_push(args.output_dir, args.iterations, args.check)
    if args.task in ("push_soft", "all"):
        task_push_soft(args.output_dir, args.iterations, args.check)
    if args.task in ("push_multi", "all"):
        task_push_multi(args.output_dir, args.iterations, args.check)
    if args.task in ("grasp", "all"):
        task_grasp(args.output_dir, args.iterations, args.check)
    if args.task in ("grasp_soft", "all"):
        task_grasp_soft(args.output_dir, args.iterations, args.check)
    if args.task in ("tendon", "all"):
        task_tendon(args.output_dir, args.iterations, args.check)
    if args.task in ("haul", "all"):
        task_haul(args.output_dir, args.iterations, args.check)
    print(f"done in {time.time() - start:.1f} s")
    physics.shutdown()


if __name__ == "__main__":
    main()
