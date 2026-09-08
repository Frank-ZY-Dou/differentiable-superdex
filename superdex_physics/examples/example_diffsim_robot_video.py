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
   ground with friction). A running cost on the cube's orientation keeps the
   push flat: without it the optimizer shoves the cube over onto an edge.
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
``robot_push_policy.mp4`` (``--task push_policy``) is a straight push solved by a
feedback policy on top of an open-loop plan: a small MLP maps the observed arm
pose, cube position and orientation and the time (normalized around the initial
observation) to a
residual on the plan's joint targets, and its weights are trained with the
gradient through the simulator (``PolicyRollout``: analytic policy gradients,
feedback path included) jointly on three cube starts, the nominal one the plan
was optimized on and two 1.5 cm to either side (Adam; a step that loses the
contact is undone and the learning rate halved). The loss is the cube's final
distance to the goal plus a per-step penalty on its rotation: an off-center push
spins the cube and a spun cube's final position is a chaotic function of the
push. An open-loop residual of the same architecture, trained the same way, is
the baseline: one trajectory cannot serve all three starts, the feedback policy
re-centers on the observed cube; the video ends with the plan alone and both
policies on every start.

``robot_hand_grasp.mp4`` (``--task hand``) is the grasp with a five-finger hand
(FR3 + Tesollo DG-5F): the hand comes down above and behind a 5 cm cube with the
fingers horizontal, pinches it between the fingertips of fingers 2-4 on its far
face and the thumb on its near face, lifts it, and the carry knots are optimized
like the gripper grasp (27 controlled DoFs, the finger pads in frictional contact
at the engine's default stiffness; see ``DG5F_GRASP`` for the geometry and why a
wrap is not reachable). The same pinch runs with the other hands of the
``HAND_GRASPS`` profiles: ``--task wuji2`` (Wuji Hand 2, beta 1, the SuperDex
asset), ``wuji2b2`` (Wuji Hand 2, beta 2), ``wuji1`` (Wuji Hand 1), ``xhand``
(XHand1) and ``sharpa`` (Sharpa Wave); their hand packages under
``assets/bots/hands`` and the FR3 + hand recipes under
``assets/bots/arm_hand_combos`` are part of this fork.

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
import dataclasses
import functools
import os
import pathlib
import sys
import time

os.environ.setdefault("SUPERDEX_PRECISION", "double")

import numpy as np
import superdex.physics as physics
import superdex.robotics as robotics
import torch
from superdex.physics.diffsim_rollout import ForwardSolveError, step_with_substeps
from superdex.physics.diffsim_torch import (
    ArticulatedPoseObservation,
    OrientationObservation,
    PolicyRollout,
    TorchRollout,
    TranslationObservation,
)
from superdex.physics.paths import resolve_asset, resolve_asset_root
from superdex.physics.utils import render_model_registry
from superdex.physics.utils.penetration import PenetrationChecker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import example_diffsim_video  # noqa: E402
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
# The engine's default stiffness with a higher friction (see CONTACT): at 1e6 the fingertip pads
# sank 15 mm into the 5 cm cube, at 1e9 3.4 mm; the gradient checks are alike (along the
# gradient 1.4e-4, the stiff closed-loop island makes single entries rough either way).
GRASP_CONTACT = physics.ContactParams(penalty_coefficient=1e9, coulomb_friction_coefficient=0.8)
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
# With the box's 768 contact triangles (cube_shape) the island's residual floor sits at ~2.5e-6
# on some steps; stalls below this are accepted. Under aggressive trajectories a step of the
# cable island occasionally runs out of Newton iterations higher up (a slack, buckling cable);
# such steps are substepped.
HAUL_STALL_TOLERANCE = 5e-6
HAUL_SUBSTEP_LEVELS = 2
HAUL_LEARNING_RATE = 0.004  # Adam; 0.002 halves the loss in 40 iterations, 0.004 reaches 0.00096 in 120
# Five-finger grasps: a HandGrasp profile per hand (HAND_GRASPS). All of them pinch the same
# 5 cm cube (HAND_CUBE_*) on the same timeline: the hand comes down with open fingers, closes
# them over HAND_CLOSE_STEPS steps from HAND_CLOSE_START and lifts; the carry knots from
# HAND_CARRY_START on are the optimized controls. The hand is placed by the engine's IK with a
# position and a rotation target on the profile's wrist link (one of each per link; the IK
# solves from the current pose, and the arm's flange angle at the start decides which branch
# it converges to). A pinch squeezes the position-controlled fingertips a few millimetres into
# the cube at the default contact stiffness; that is the grip force, and the replays'
# penetration limit allows it. The hand-cube islands (27 DoFs, some 20 finger links in
# contact) occasionally run out of Newton iterations at ~5e-7 residual: stalls below the
# profile's tolerance are accepted, worse steps are substepped.
HAND_CUBE_HALF = 0.025
HAND_CUBE_DENSITY = 300.0  # [kg/m^3]: a 5 cm cube of 0.04 kg
HAND_CUBE_POS = np.array([0.45, 0.0, HAND_CUBE_HALF - 0.001])
HAND_CLOSE_START, HAND_CLOSE_STEPS, HAND_LIFT_END = 30, 25, 100
HAND_CARRY_START = 65


@dataclasses.dataclass(frozen=True)
class HandGrasp:
    """The pinch of one five-finger hand on the cube."""

    name: str  # task name: the video is robot_<name>_grasp.mp4
    title: str  # e.g. "FR3 + DG-5F five-finger grasp"
    bot: str  # the arm-hand recipe (an asset path)
    wrist_link: str  # the link the IK places (a name suffix)
    rotation: np.ndarray  # columns: that link's x, y, z axes in the world at the grasp
    flange_start: float  # [rad] the FR3's joint 7 at the start of the IK
    offset: tuple  # [m] the wrist link's origin from the cube center at the grasp (world x, y, z)
    fingers: dict  # closed targets of the finger joints, by joint name (the rest keep the default pose)
    thumb: dict  # closed targets of the thumb joints
    lifted_cube: tuple  # [m] where the lifted cube ends on the initial trajectory (the goal's base)
    thumb_pre: dict | None = None  # a pose the thumb goes through first (None: straight to closed)
    thumb_swing: float = 0.6  # the fraction of the closure spent reaching thumb_pre
    open: dict = dataclasses.field(default_factory=dict)  # joints held away from the default pose while open
    max_penetration: float = 0.01  # [m] the interpenetration the replays may reach
    stall_tolerance: float = 1e-5  # the Newton residual below which a stalled solve is accepted


# Tesollo DG-5F (short wrist), the hand shipped with SuperDex. Palm frame: local x is the palm
# normal, local z the finger direction, local y across the fingers. The hand is placed with the
# fingers horizontal along +y and the palm facing down (local x -> -z, local z -> +y); fingers
# 2-5 flex on their three flexion joints (x_2 MCP, x_3 PIP, x_4 DIP) and the thumb goes to
# joints 1_1 (abduction), 1_2 (opposition), 1_3, 1_4 as given. In the palm frame the closed
# fingertips sit 90 mm below and 135 mm ahead of the palm center and the thumb tip 90 mm below
# and 80 mm ahead: the palm goes 95 mm above and 110 mm behind the cube center, 15 mm to the
# side so that fingers 2 and 3 straddle the cube's lateral center; fingers 2-4 curl onto the
# cube's far face at its mid-height and the thumb, opposed under the palm, presses the near
# face. A power wrap is not reachable for a cube this size: the thumb's tip cannot get more
# than ~50 mm behind the finger pads along the finger direction (its opposition sweeps
# laterally), so palm-down wraps of a 7 cm cube only "held" it by passing the fingers through
# it at a compliant contact. The IK started at the arm's default flange angle (pi/2) runs
# into that joint's upper limit for this pose; started at 0 it converges (joint 7 near -2).
DG5F_GRASP = HandGrasp(
    name="hand",
    title="FR3 + DG-5F five-finger grasp",
    bot="bots/arm_hand_combos/fr3_dg5f_short/right/fr3_dg5f_short_right.superdex_bot",
    wrist_link="dg5f_link_palm",
    rotation=np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).T,
    flange_start=0.0,
    offset=(0.015, -0.11, 0.095),
    fingers={
        f"dg5f_joint_{finger}_{joint}": amount
        for finger in (2, 3, 4, 5)
        for joint, amount in zip((2, 3, 4), (0.5, 1.0, 0.4))
    },
    thumb={"dg5f_joint_1_1": 0.3, "dg5f_joint_1_2": -1.2, "dg5f_joint_1_3": 1.0, "dg5f_joint_1_4": 0.0},
    lifted_cube=(0.464, -0.019, 0.271),
)

# Wuji Hand 2 (beta 1, the SuperDex asset; beta 2 below is this fork's conversion of the
# newer description, with the same joint names and kinematics). In the wrist frame the fingers
# extend along -z and curl toward +y (the palm side), the thumb rests on the +x side with its
# pulp toward the fingers. Palm down with the fingers along +y (wrist x -> -X, y -> -Z,
# z -> -Y); the arm sags 10 mm under the hand at these gains. With the closed targets the
# index and middle pads sit 120 mm ahead of and 48 mm below the wrist, their pulps turned
# back toward the wrist, and the thumb pad 73 mm ahead of and 61 mm below it, its pulp
# toward the fingertips: the cube, centered 95 mm ahead and 70 mm below the wrist, is pinched
# between the thumb on its near face and fingers 2-3 on its far face, the pulps' closed
# positions about a centimetre inside the faces; fingers 4-5 close beside the cube. The thumb
# closes in two moves: it first swings across the palm (thumb_pre), its pulp turned toward
# the fingertips and still in front of the cube's near face, then advances along the fingers;
# a straight interpolation to the closed pose sweeps it through the cube's near corner and
# pushes the cube away.
_WUJI2_FINGERS = {
    f"r_{finger}_{joint}": amount
    for finger in ("index_finger", "middle_finger", "ring_finger", "pinky")
    for joint, amount in zip(("mcp_flex", "pip", "dip"), (0.5, 1.0, 0.4))
}
_WUJI2_THUMB_PRE = {"r_thumb_cmc_flex": 1.241, "r_thumb_cmc_abd": -0.282, "r_thumb_mcp": -0.997, "r_thumb_ip": -0.229}
_WUJI2_THUMB = {"r_thumb_cmc_flex": 1.241, "r_thumb_cmc_abd": -0.139, "r_thumb_mcp": -0.519, "r_thumb_ip": -0.78}
_PALM_DOWN_FINGERS_FORWARD = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]]).T
WUJI2_GRASP = HandGrasp(
    name="wuji2",
    title="FR3 + Wuji Hand 2 grasp",
    bot="bots/arm_hand_combos/fr3_wuji_hand2_beta1/right/fr3_wuji_hand2_beta1_right.superdex_bot",
    wrist_link="r_wrist",
    rotation=_PALM_DOWN_FINGERS_FORWARD,
    flange_start=np.pi / 2,
    offset=(0.018, -0.095, 0.080),
    fingers=_WUJI2_FINGERS,
    thumb=_WUJI2_THUMB,
    thumb_pre=_WUJI2_THUMB_PRE,
    lifted_cube=(0.460, -0.002, 0.270),
)
WUJI2B2_GRASP = dataclasses.replace(
    WUJI2_GRASP,
    name="wuji2b2",
    title="FR3 + Wuji Hand 2 (beta 2) grasp",
    bot="bots/arm_hand_combos/fr3_wuji_hand2_beta2/right/fr3_wuji_hand2_beta2_right.superdex_bot",
    lifted_cube=(0.461, -0.002, 0.269),
)

# Wuji Hand 1. In the palm frame the fingers extend along +z and curl toward +x (the palm
# side), the thumb (finger 1) rests on the +y side. Palm down with the fingers along +y
# (palm x -> -Z, y -> -X, z -> +Y); the cube is centered 110 mm ahead of and 65 mm below the
# palm, 5 mm toward the little finger. Fingers 2-5 flex on joints 1, 3 and 4 (joint 2 abducts):
# closed, fingers 2 and 3 reach 8 mm past the cube's far face. The thumb descends already
# curled above the cube's near-top edge (open) and unrolls onto the near face (thumb, its tip
# 6 mm inside the face); closing it from the straight pose instead swings it through the
# cube's place and below the palm, into the table.
WUJI1_GRASP = HandGrasp(
    name="wuji1",
    title="FR3 + Wuji Hand 1 grasp",
    bot="bots/arm_hand_combos/fr3_wuji_hand1/right/fr3_wuji_hand1_right.superdex_bot",
    wrist_link="palm_link",
    rotation=np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).T,
    flange_start=0.0,
    offset=(-0.005, -0.11, 0.075),
    fingers={
        f"finger{finger}_joint{joint}": amount
        for finger in (2, 3, 4, 5)
        for joint, amount in zip((1, 3, 4), (0.6, 1.0, 0.4))
    },
    thumb={"finger1_joint1": 1.2, "finger1_joint2": -0.6, "finger1_joint3": 0.85, "finger1_joint4": 0.85},
    open={"finger1_joint1": 1.6, "finger1_joint2": -0.3, "finger1_joint3": 1.1, "finger1_joint4": 1.1},
    lifted_cube=(0.446, 0.000, 0.270),
)

# XHand1. In the root frame (the wrist mount) the fingers extend along -z and curl toward +y
# (the palm side), the thumb rests on the +x side. Palm down with the fingers along +y as for
# the Wuji Hand 2; the cube is centered 135 mm ahead of and 70 mm below the root, 2 cm under
# the palm. Each finger flexes on its two joints: closed, the fingertips reach 7 mm past the
# far face. The thumb is 12 cm long and would touch the table while bending across the palm
# at this height, so the hand descends with it already bent across and fully flexed (open),
# just in front of the cube's near face, and the closure unrolls it onto that face (thumb).
# The thumb presses the near face lower than the fingers press the far face, so the cube
# rocks a few degrees at closure and settles tilted in the lifted grip; a less unrolled thumb
# (rota joint 2 at 1.25 rather than 0.9) keeps that within about 25 degrees.
XHAND_GRASP = HandGrasp(
    name="xhand",
    title="FR3 + XHand1 grasp",
    bot="bots/arm_hand_combos/fr3_xhand1/right/fr3_xhand1_right.superdex_bot",
    wrist_link="right_hand_link",
    rotation=_PALM_DOWN_FINGERS_FORWARD,
    flange_start=np.pi / 2,
    offset=(0.015, -0.135, 0.08),
    fingers={
        f"right_hand_{finger}_joint{joint}": amount
        for finger in ("index", "mid", "ring", "pinky")
        for joint, amount in zip((1, 2), (0.6, 1.0))
    },
    thumb={"right_hand_thumb_bend_joint": 1.4, "right_hand_thumb_rota_joint1": 0.8, "right_hand_thumb_rota_joint2": 1.25},
    open={"right_hand_thumb_bend_joint": 1.4, "right_hand_thumb_rota_joint1": 0.8, "right_hand_thumb_rota_joint2": 1.6},
    lifted_cube=(0.453, -0.001, 0.250),
)

# Sharpa Wave (right). In the frame of the hand's base link (right_hand_C_MC) the fingers
# extend along +z and curl toward +x (the palm side), the thumb rests on the +y side: the same
# placement as the Wuji Hand 1 (base x -> -Z, y -> -X, z -> +Y). The cube is centered 105 mm
# ahead of and 75 mm below the base, 10 mm toward the thumb side, so that the index, middle
# and ring fingertips (30, 10 and -10 mm across the hand) land on its far face; each finger
# flexes on its MCP, PIP and DIP joints and, closed, the fingertips reach 6-9 mm past the far
# face at the cube's mid-height. The thumb descends bent across the palm and fully flexed
# (open), its tip 16 mm in front of the cube's near face, and the closure unrolls its
# interphalangeal joint onto that face (tip 8 mm inside, at the mid-height); closing the
# straight thumb instead sweeps it in from the side through the cube's place.
_SHARPA_THUMB_OPEN = {
    "right_thumb_CMC_FE": 1.06,
    "right_thumb_CMC_AA": -0.3,
    "right_thumb_MCP_FE": 1.35,
    "right_thumb_MCP_AA": -0.3,
    "right_thumb_IP": 1.65,
}
SHARPA_GRASP = HandGrasp(
    name="sharpa",
    title="FR3 + Sharpa Wave grasp",
    bot="bots/arm_hand_combos/fr3_sharpa_wave/right/fr3_sharpa_wave_right.superdex_bot",
    wrist_link="right_hand_C_MC",
    rotation=np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).T,
    flange_start=0.0,
    offset=(0.01, -0.105, 0.075),
    fingers={
        f"right_{finger}_{joint}": amount
        for finger in ("index", "middle", "ring", "pinky")
        for joint, amount in zip(("MCP_FE", "PIP", "DIP"), (0.6, 1.0, 0.5))
    },
    thumb={**_SHARPA_THUMB_OPEN, "right_thumb_IP": 0.33},
    open=_SHARPA_THUMB_OPEN,
    lifted_cube=(0.453, 0.005, 0.264),
)

HAND_GRASPS = {p.name: p for p in (DG5F_GRASP, WUJI2_GRASP, WUJI2B2_GRASP, WUJI1_GRASP, XHAND_GRASP, SHARPA_GRASP)}
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
PUSH_SOFT_PENALTY = 1e6  # [Pa/m] contact stiffness of the soft push (cube, arm, ground); see _task_push_impl
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
# The engine's default contact stiffness (1e9 Pa/m). A 1e6 material, kept from the first
# gradient checks, let the wrist links sink ~1 cm into the pushed cube and the cube 3.5 mm into
# the ground (a penalty contact is compliant, and the joint controller pushes on); at 1e9 the
# penetration is ~1 mm, the forward Newton solve needs fewer iterations (40 vs 65 on average)
# and the adjoint's gradient check is as clean (entries agree with FD to 1e-6..2e-5).
CONTACT = physics.ContactParams(penalty_coefficient=1e9, coulomb_friction_coefficient=0.4)
# The interpenetration the replays of the rigid tasks may reach [m] (the deepest contact
# sample of any actor pair, checked by ConvergenceGuard): a penalty contact at the default
# stiffness overlaps by 1-3.5 mm under these loads (the wrist on the pushed cube, the
# gripper pads on the grasped one); more means a softer material or a broken contact.
MAX_PENETRATION = 0.005
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


# Contact in SuperDex acts at sample points of one body's surface tested against the other
# body's distance field; the sample points of a triangle mesh lie inside its triangles. A cube
# of 12 triangles (cube_coords) has no sample near its edges and corners, so a tilted cube
# sinks a corner 10 mm into the ground before any sample sees it. The rigid cubes therefore
# use a structured mesh with cells of about CUBE_CELL_SIZE, whose samples sit within a few
# millimetres of every edge and corner (8 cells per edge on the 10 cm cubes, 4 on the 5 cm
# ones; a 5 cm cube of 8 cells put so many samples between the fingertips of a five-finger
# hand that the Newton solve of the lift failed).
CUBE_CELL_SIZE = 0.0125  # [m]


def cube_shape(half: float):
    """The tet-mesh shape of a cube of half size ``half`` with cells of about CUBE_CELL_SIZE."""
    cells = max(1, int(round(2.0 * half / CUBE_CELL_SIZE)))
    coordinates, connectivity = box_tet_mesh(size=2.0 * half, cells=cells)
    return physics.create_tet_mesh_shape(coordinates=coordinates, connectivity=connectivity)


class CubeOrientationLoss:
    """0.5 * PUSH_YAW_WEIGHT * ||q - q0||^2 on a rigid actor's orientation quaternion at every
    step of a push (q0 is the orientation at construction): the push keeps the cube flat and
    square, see PUSH_YAW_WEIGHT."""

    def __init__(self, cube):
        self.cube = cube
        self.quat_ref = np.asarray(cube.get_center_of_mass_transform().rotation.tolist(), dtype=np.float64)

    def _diff(self) -> np.ndarray:
        return np.asarray(self.cube.get_center_of_mass_transform().rotation.tolist(), dtype=np.float64) - self.quat_ref

    def value(self) -> float:
        d = self._diff()
        return 0.5 * PUSH_YAW_WEIGHT * float(d @ d)

    def accumulate_output_grad(self) -> None:
        g = np.zeros(7)
        g[3:] = PUSH_YAW_WEIGHT * self._diff()
        diffsim.get_center_of_mass_transform_backward(self.cube, g)


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


def spawn_arm(scene, with_controller: bool = True, with_contact: bool = True, contact=None):
    """FR3 arm; see :func:`spawn_bot`."""
    return spawn_bot(scene, ARM_BOT, EE_LINK, _arm_gains(), with_controller, with_contact, contact)


def spawn_gripper_arm(
    scene, with_controller: bool = True, with_contact: bool = True, link_shape_files=None, contact=None
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
        GRASP_CONTACT if contact is None else contact,
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


def ik_joint_poses(waypoints, spawner=None, link_name=None):
    """Joint poses reaching the waypoints with the end-effector link (or the link whose
    name ends with ``link_name``), from a dedicated zero-gravity IK scene (the engine
    solves IK as a quasi-static simulation). ``spawner`` builds the bot (default: the
    FR3 arm)."""
    spawner = spawn_arm if spawner is None else spawner
    scene = physics.create_scene("ik")
    scene.set_gravity([0.0, 0.0, 0.0])
    bot, arm, ee, context = spawner(scene, with_controller=False, with_contact=False)
    if link_name is not None:
        links = []
        scene.for_each_actor(lambda a: links.append(a) if a.is_nested_link_actor() else None)
        ee = [a for a in links if a.get_name().endswith(link_name)][0]
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
    reported rather than treated as a failure. Anything worse aborts the run.

    It also records the interpenetration of every replayed step (the deepest contact
    sample per actor pair, ``superdex.physics.utils.penetration``); ``max_penetration``
    [m] makes a deeper overlap abort the run at the end (None: report only, for the
    tasks whose compliant contact is a known issue)."""

    stall_tolerance = 1e-7

    def __init__(self, scene, stall_tolerance: float | None = None, max_penetration: float | None = None):
        self.scene = scene
        if stall_tolerance is not None:
            self.stall_tolerance = stall_tolerance
        self.max_penetration = max_penetration
        self.penetration = PenetrationChecker(scene)
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
        self.penetration.record(self.checked)
        self.checked += 1

    def report(self, name: str) -> None:
        print(
            f"[{name}] forward convergence verified on {self.checked} replayed steps "
            f"({self.stalled} stalled below {self.stall_tolerance:.0e} residual, "
            f"{len(self.splits)} split into substeps: {self.splits[:8]})"
        )
        print(f"[{name}] " + self.penetration.report(self.max_penetration).replace("\n", f"\n[{name}] "))
        if self.max_penetration is not None:
            self.penetration.assert_below(self.max_penetration)


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
    max_penetration: float | None = MAX_PENETRATION,
    step_losses=None,
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
    overrides the guard's default residual floor for stalled solves;
    ``max_penetration`` the interpenetration the replays may reach (see
    :class:`ConvergenceGuard`); ``step_losses`` (``step -> [loss]``) adds running
    costs to the terminal ``loss``."""
    num_steps = controls0.shape[0]
    guard = ConvergenceGuard(scene, stall_tolerance, max_penetration)
    bridge = TorchRollout(
        scene,
        dt=dt,
        num_steps=num_steps,
        control_actors=[arm],
        terminal_losses=[loss],
        step_losses=step_losses,
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


class ResidualPolicy(torch.nn.Module):
    """Targets = initial open-loop trajectory at the current step + a small MLP residual. With
    ``feedback`` the residual is a function of the observation (arm pose, cube position) and
    the normalized time; without it, of the time alone: an open-loop residual with the same
    architecture and optimizer, the baseline the feedback policy is compared against. The
    residual head starts at zero, so iteration 0 reproduces the open-loop rollout."""

    def __init__(self, controls0: np.ndarray, features0: np.ndarray, feature_scale: np.ndarray, feedback: bool):
        """``features0`` / ``feature_scale``: the input features (observations then the time)
        are fed as ``(features - features0) / feature_scale``, of order one over a push, so
        one learning rate suits the joint angles, the cube position and the time alike."""
        super().__init__()
        self.register_buffer("controls0", torch.tensor(controls0, dtype=torch.float64))
        self.register_buffer("features0", torch.tensor(features0, dtype=torch.float64))
        self.register_buffer("feature_scale", torch.tensor(feature_scale, dtype=torch.float64))
        self.num_steps = controls0.shape[0]
        self.feedback = feedback
        self.net = torch.nn.Sequential(
            torch.nn.Linear((features0.shape[0] - 1 if feedback else 0) + 1, PUSH_POLICY_HIDDEN),
            torch.nn.Tanh(),
            torch.nn.Linear(PUSH_POLICY_HIDDEN, PUSH_POLICY_HIDDEN),
            torch.nn.Tanh(),
            torch.nn.Linear(PUSH_POLICY_HIDDEN, controls0.shape[1]),
        ).double()
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        step = int(round(float(features[-1].detach()) * self.num_steps))
        normalized = (features - self.features0) / self.feature_scale
        inputs = normalized if self.feedback else normalized[-1:]
        return self.controls0[min(step, self.num_steps - 1)] + PUSH_POLICY_RESIDUAL_SCALE * self.net(inputs)


def optimize_open_loop_targets(
    scene, arm, loss, step_losses, controls0: np.ndarray, dt: float, num_iterations: int, learning_rate: float, grad_clip: float
) -> np.ndarray:
    """Adam on the per-step joint targets (as ``run_task`` does, without the recording):
    returns the optimized targets; the scene is left at the end of the last rollout."""
    bridge = TorchRollout(
        scene, dt=dt, num_steps=controls0.shape[0], control_actors=[arm], terminal_losses=[loss], step_losses=step_losses
    )
    try:
        params = torch.tensor(controls0, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([params], lr=learning_rate)
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            value = bridge(controls=params)
            value.backward()
            torch.nn.utils.clip_grad_norm_([params], max_norm=grad_clip)
            optimizer.step()
            if iteration % 10 == 0 or iteration == num_iterations - 1:
                print(f"  open-loop plan iter {iteration:3d}  loss {float(value.detach()):.6f}")
        return params.detach().numpy().copy()
    finally:
        bridge.close()


def task_push_policy(output_dir: pathlib.Path, num_iterations: int, check: bool) -> None:
    """The push with a feedback policy (see the module docstring). The open-loop plan is first
    optimized on the nominal start (``optimize_open_loop_targets``); the residual MLP on top of
    it is then trained jointly on the ``PUSH_POLICY_STARTS`` cube starts by Adam (learning
    rate decaying linearly to a tenth) with the gradient clipped and worsening steps undone
    (a step that raises the mean loss by more than ``PUSH_POLICY_MAX_INCREASE`` is reverted:
    losing the contact lands on the zero-gradient plateau of the untouched cube, from which no
    gradient recovers, and near its best the policy's smooth-branch gradient no longer
    describes the rough landscape - unchecked, Adam then climbs steadily). The piecewise-smooth loss of frictional contact - jumps of ~1e-7 between Newton
    solution branches, and a chaotic sensitivity of the off-center starts - makes an Armijo
    line search stall; Adam's fixed-size steps step over it. An open-loop residual of the same
    architecture, trained the same way, is the baseline: one trajectory cannot serve all
    starts, the feedback policy reacts to the observed cube."""
    name = "robot_push_policy"
    cube_start = np.array([0.55, 0.0, CUBE_HALF - 0.001])
    goal = PUSH_POLICY_GOAL
    print(f"[{name}] IK for the initial joint-target trajectory")
    waypoints = [np.array([0.44 + 0.05 * k, 0.0, 0.10]) for k in range(6)]
    poses = ik_joint_poses(waypoints)
    num_steps = 75
    keys = [(0, poses[0])] + [(15 * k, poses[k]) for k in range(1, 6)]
    controls_ik = interpolate_targets(keys, num_steps)

    scene = physics.create_scene("Differentiable push (policy)")
    scene.set_gravity(GRAVITY)
    bot, arm, ee, context = spawn_arm(scene)
    scene.create_rigid_actor(
        name="ground",
        shape=physics.create_plane_shape(normal=[0, 0, 1], distance=0.0),
        is_static=True,
        contact=CONTACT,
    )
    cube = scene.create_rigid_actor(
        name="cube",
        shape=cube_shape(CUBE_HALF),
        density=300.0,
        contact=CONTACT,
        world_from_local=physics.TransformRT(cube_start.tolist()),
    )
    arm.set_articulated_pose_from_joints(poses[0])
    configure(scene)
    dt = 0.02
    cube_position = lambda: np.asarray(cube.get_center_of_mass_transform().translation)

    class CubeLoss:
        def value(self) -> float:
            d = cube_position() - goal
            return 0.5 * float(d @ d)

        def accumulate_output_grad(self) -> None:
            g = np.zeros(7)
            g[:3] = cube_position() - goal
            diffsim.get_center_of_mass_transform_backward(cube, g)

    yaw_loss = CubeOrientationLoss(cube)
    step_losses = lambda step: [yaw_loss]
    observations = [ArticulatedPoseObservation(arm), TranslationObservation(cube), OrientationObservation(cube)]
    state_nominal = scene.capture_state()

    def place_cube(offset) -> None:
        scene.restore_state(state_nominal, False)
        cube.set_center_of_mass_transform(
            physics.TransformRT((cube_start + np.array([offset[0], offset[1], 0.0])).tolist())
        )

    kinds = ("open-loop", "feedback")  # the feedback policy last: its iterations are recorded
    policies: dict[str, ResidualPolicy] = {}
    rollouts: dict[str, list[PolicyRollout]] = {}
    recorder = None
    try:
        print(f"[{name}] open-loop plan: Adam on the joint targets, nominal start")
        controls_plan = optimize_open_loop_targets(
            scene, arm, CubeLoss(), step_losses, controls_ik, dt, PUSH_POLICY_PLAN_ITERATIONS, 0.002, 0.02
        )
        place_cube(PUSH_POLICY_STARTS[0])
        features0 = np.concatenate([o.value() for o in observations] + [[0.0]])
        feature_scale = np.concatenate([np.full(arm.get_num_dofs(), 0.3), np.full(3, 0.25), np.full(4, 0.2), [1.0]])
        for kind in kinds:
            torch.manual_seed(0)
            policies[kind] = ResidualPolicy(controls_plan, features0, feature_scale, kind == "feedback")
            rollouts[kind] = []
            for offset in PUSH_POLICY_STARTS:
                place_cube(offset)
                rollouts[kind].append(
                    PolicyRollout(
                        scene,
                        dt=dt,
                        num_steps=num_steps,
                        policy=policies[kind],
                        observations=observations,
                        control_actors=[arm],
                        time_feature=True,
                        terminal_losses=[CubeLoss()],
                        step_losses=step_losses,
                        max_substep_levels=PUSH_SUBSTEP_LEVELS,
                        substep_residual_tolerance=PUSH_POLICY_STALL_TOLERANCE,
                    )
                )
        guard = ConvergenceGuard(scene, PUSH_POLICY_STALL_TOLERANCE, MAX_PENETRATION)
        recorder = Recorder(
            scene,
            goal,
            look_from=[1.9, -1.6, 0.9],
            look_at=[0.6, 0.05, 0.1],
            title="FR3 push with a feedback policy: an MLP trained through the simulator",
        )

        def replay(rollout: PolicyRollout, policy, capture: bool, caption_fn=None) -> float:
            """The closed loop as the rollout runs it (same substepping); ``policy=None`` replays
            the open-loop plan alone. Returns the loss."""
            scene.restore_state(rollout.initial_state, False)
            if capture:
                recorder.begin_iteration()
            total = 0.0
            for step in range(num_steps):
                if policy is None:
                    targets = controls_plan[step]
                else:
                    features = np.concatenate([o.value() for o in observations] + [[step / num_steps]])
                    with torch.no_grad():
                        targets = policy(torch.tensor(features, dtype=torch.float64)).numpy()
                arm.set_articulated_target_pose(np.ascontiguousarray(targets))
                guard.note_substeps(
                    step, step_with_substeps(scene, dt, PUSH_SUBSTEP_LEVELS, guard.stall_tolerance, step=step)
                )
                guard.assert_last_step()
                total += yaw_loss.value()
                if capture:
                    recorder.capture(cube_position(), caption_fn(step), hold=(1 if step < num_steps - 1 else 12))
            return total + CubeLoss().value()

        def closed_loop_loss(kind: str) -> float:
            return float(np.mean([replay(r, policies[kind], capture=False) for r in rollouts[kind]]))

        losses = {kind: [] for kind in kinds}
        best = {}  # kind -> (iteration, mean loss, parameters): the policy the comparison uses
        record = {0, 1, 2, 4, 7, 12, 20, 30, num_iterations - 1}
        for kind in kinds:
            policy = policies[kind]
            params = list(policy.parameters())
            optimizer = torch.optim.Adam(params, lr=PUSH_POLICY_LEARNING_RATE)
            for iteration in range(num_iterations):
                # Linear decay of the learning rate to a tenth over the run.
                optimizer.param_groups[0]["lr"] = PUSH_POLICY_LEARNING_RATE * (1.0 - 0.9 * iteration / max(num_iterations - 1, 1))
                optimizer.zero_grad()
                loss = sum(r() for r in rollouts[kind]) / len(rollouts[kind])
                loss.backward()
                loss_value = float(loss.detach())
                losses[kind].append(loss_value)
                if kind not in best or loss_value < best[kind][1]:
                    best[kind] = (iteration, loss_value, [p.detach().clone() for p in params])
                grads = [p.grad.detach().clone() for p in params]
                grad_norm = float(torch.sqrt(sum((g * g).sum() for g in grads)))
                results = [r.last_result for r in rollouts[kind]]
                if kind == "feedback" and iteration == 0 and check:
                    # Directional derivative along the gradient: central finite differences of
                    # every start's closed-loop loss with the policy weights perturbed by eps =
                    # 1e-6, 1e-7, 1e-8 along the normalized gradient of the mean loss. The
                    # frictional contact makes the loss piecewise smooth: at steps where the
                    # forward Newton solve is nearly degenerate (100+ iterations to 1e-9) a
                    # perturbation of a few 1e-8 can land it on another local solution, a jump
                    # of ~1e-7 in the loss (5e-6 m in the cube position). The adjoint is the
                    # exact derivative of the branch taken, so the comparison holds for the
                    # starts whose three FD values agree; the check reports the others as jumps.
                    # It is only meaningful if the perturbed replays take the same substeps.
                    splits_before = len(guard.splits)
                    direction = [g / grad_norm for g in grads]
                    epsilons = (1e-6, 1e-7, 1e-8)
                    fd = np.zeros((len(rollouts[kind]), len(epsilons)))
                    for j, eps in enumerate(epsilons):
                        for sign_index, sign in enumerate((+1.0, -1.0)):
                            with torch.no_grad():
                                for p, d in zip(params, direction):
                                    p.add_(sign * eps * d)
                            for i, r in enumerate(rollouts[kind]):
                                fd[i, j] += (1.0 if sign_index == 0 else -1.0) * replay(r, policy, capture=False)
                            with torch.no_grad():
                                for p, d in zip(params, direction):
                                    p.sub_(sign * eps * d)
                        fd[:, j] /= 2.0 * eps
                    print(f"  gradient check (adjoint vs closed-loop central FD at eps 1e-6 / 1e-7 / 1e-8):")
                    for i, offset in enumerate(PUSH_POLICY_STARTS):
                        spread = (fd[i].max() - fd[i].min()) / max(abs(fd[i]).max(), 1e-300)
                        print(
                            f"    start offset y = {100 * offset[1]:+.1f} cm: fd "
                            + " / ".join(f"{v:+.5e}" for v in fd[i])
                            + (
                                "  [smooth]"
                                if spread < 1e-2
                                else "  [loss jumps within the perturbation: another Newton solution branch]"
                            )
                        )
                    mean_fd = fd.mean(axis=0)  # eps 1e-7: below the jumps, above the solver noise
                    print(
                        f"    mean: |grad| {grad_norm:.5e}  fd(1e-7) {mean_fd[1]:+.5e}  "
                        f"rel err {abs(mean_fd[1] - grad_norm) / grad_norm:.1e}"
                        + (
                            f"  [{len(guard.splits) - splits_before} perturbed replay steps were split into "
                            "substeps (a different discretization): the comparison is approximate]"
                            if len(guard.splits) > splits_before
                            else ""
                        )
                    )
                if kind == "feedback" and iteration in record:
                    replay(
                        rollouts[kind][0],
                        policy,
                        capture=True,
                        caption_fn=lambda step, it=iteration: (
                            f"iteration {it}   t = {(step + 1) * dt:.2f} s   mean loss = {losses['feedback'][-1]:.5f}"
                        ),
                    )
                # Adam with the gradient clipped; a step that raises the mean closed-loop loss
                # by more than PUSH_POLICY_MAX_INCREASE is undone (the moments keep the
                # gradient, the next step is shorter by the decay). The tolerance lets the
                # optimizer step over the loss's small jumps; it refuses a lost contact, and the
                # steady climb of the late iterations when the smooth-branch gradient stops
                # describing the rough landscape (a policy already near its best).
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                before = [p.detach().clone() for p in params]
                optimizer.step()
                try:
                    trial_loss = closed_loop_loss(kind)
                    rejected = trial_loss > loss_value * (1.0 + PUSH_POLICY_MAX_INCREASE)
                    reason = f"loss x{trial_loss / loss_value:.1f}"
                except ForwardSolveError as error:
                    # A violent closed loop the forward solve cannot follow even substepped:
                    # the same kind of step as a lost contact.
                    rejected = True
                    reason = f"forward solve failed at step {error.step}, dt {error.dt:g}, residual {error.residual_norm:.1e}"
                if rejected:
                    with torch.no_grad():
                        for p, b in zip(params, before):
                            p.copy_(b)
                splits = [(i, r.split_steps) for i, r in enumerate(results) if r.split_steps]
                print(
                    f"[{name}] {kind:9s} iter {iteration:3d}  loss {loss_value:.6f}  |grad| {grad_norm:.3e}  "
                    f"lr {optimizer.param_groups[0]['lr']:.1e}  adjoint residual {max(r.max_adjoint_residual for r in results):.1e}"
                    + (f"  step rejected ({reason})" if rejected else "")
                    + (f"  substepped {splits}" if splits else "")
                )
        guard.report(name)

        # The plan alone and the trained policies on every start: the open-loop residual is one
        # trajectory for all starts, the feedback policy reacts to the observed cube. Each policy
        # is its best iterate by the training loss (Adam keeps exploring the rough landscape).
        for kind in kinds:
            iteration, loss_value, values = best[kind]
            with torch.no_grad():
                for p, v in zip(policies[kind].parameters(), values):
                    p.copy_(v)
            print(f"[{name}] {kind} policy: best iterate {iteration} (mean loss {loss_value:.6f})")
        print(f"[{name}] final cube distance to the goal per start (plan alone / open-loop residual / feedback)")
        for i, offset in enumerate(PUSH_POLICY_STARTS):
            distances = {}
            for kind, policy in (("plan", None), ("open-loop", policies["open-loop"]), ("feedback", policies["feedback"])):
                replay(
                    rollouts["feedback" if kind == "plan" else kind][i],
                    policy,
                    capture=True,
                    caption_fn=lambda step, kind=kind, offset=offset: (
                        (
                            "open-loop plan alone"
                            if kind == "plan"
                            else f"plan + trained {kind} residual policy"
                        )
                        + f"   cube start offset y = {100 * offset[1]:+.1f} cm   t = {(step + 1) * dt:.2f} s"
                    ),
                )
                distances[kind] = float(np.linalg.norm(cube_position() - goal))
            print(
                f"  start offset ({100 * offset[0]:+.1f}, {100 * offset[1]:+.1f}) cm: "
                f"{100 * distances['plan']:.2f} cm / {100 * distances['open-loop']:.2f} cm / "
                f"{100 * distances['feedback']:.2f} cm"
            )
        for kind in kinds:
            for r in rollouts[kind]:
                r.close()
        recorder.write(output_dir / f"{name}.mp4")  # closes the viewer
        recorder = None
        save_loss_curve(
            output_dir / f"{name}_loss.png", losses["feedback"], "FR3 push, feedback policy: mean loss vs iteration"
        )
        save_loss_curve(
            output_dir / f"{name}_open_loop_loss.png",
            losses["open-loop"],
            "FR3 push, open-loop residual baseline: mean loss vs iteration",
        )
    finally:
        if recorder is not None:
            recorder.viewer.close()  # an open offscreen viewer crashes the interpreter's exit
        scene.release_state(state_nominal)
        robotics.destroy_bot(scene, bot)
        physics.destroy_scene(scene)


PUSH_MULTI_GAP = 0.01  # [m] between the two cubes at the start
# Feedback-policy push: residual MLP on the initial trajectory, trained through the simulator.
PUSH_POLICY_HIDDEN = 32
PUSH_POLICY_RESIDUAL_SCALE = 0.05  # [rad] the residual head's output scale
PUSH_POLICY_PLAN_ITERATIONS = 40  # Adam iterations of the open-loop plan on the nominal start
PUSH_POLICY_GOAL = np.array([0.80, 0.0, CUBE_HALF])  # straight ahead: the plan is a square push
# Per-step weight of a pushed cube's orientation error (1e-2 dominated the distance term). A
# cube pushed off center spins, and a cube shoved hard tips over onto an edge; either makes
# the final position a rough function of the push, and a tipped cube rolls on its corners.
PUSH_YAW_WEIGHT = 2e-3
PUSH_POLICY_STARTS = ((0.0, 0.0), (0.0, -0.015), (0.0, 0.015))  # [m] cube start offsets trained jointly
PUSH_POLICY_LEARNING_RATE = 3e-3  # Adam on the residual MLP's weights (1e-3 barely moved the feedback policy)
PUSH_POLICY_MAX_INCREASE = 0.1  # a step raising the mean loss by more than this fraction is undone
PUSH_POLICY_STALL_TOLERANCE = 1e-5  # a feedback policy explores rougher contact than the open loop
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
    # A soft body's contact stiffness has to sit within a few decades of its material's: at
    # the engine default (1e9 Pa/m on a 1e5 Pa cube) the forward Newton solve fails hard at
    # isolated steps even substepped; the validated PUSH_SOFT_PENALTY is used by the cube,
    # the arm and the ground of the soft push (a pair combines both penalties).
    task_contact = (
        physics.ContactParams(
            penalty_coefficient=PUSH_SOFT_PENALTY,
            coulomb_friction_coefficient=CONTACT.coulomb_friction_coefficient,
        )
        if soft
        else CONTACT
    )
    bot, arm, ee, context = spawn_arm(scene, contact=task_contact)
    scene.create_rigid_actor(
        name="ground",
        shape=physics.create_plane_shape(normal=[0, 0, 1], distance=0.0),
        is_static=True,
        contact=task_contact,
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
                penalty_coefficient=PUSH_SOFT_PENALTY,
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
            shape=cube_shape(CUBE_HALF),
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
            shape=cube_shape(CUBE_HALF),
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
    # The rigid pushes keep their cubes flat and square (a running orientation cost, as in the
    # policy push); a soft cube has no orientation.
    step_losses = None
    if not soft:
        holds = [CubeOrientationLoss(cube)] + ([CubeOrientationLoss(cube2)] if multi else [])
        step_losses = lambda step: holds

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
            # The soft cube and the cube-cube impact of the two-cube push (stiff, engine-default
            # contact) each trap the forward Newton solve at isolated steps: substep them.
            substep_levels=PUSH_SUBSTEP_LEVELS if (soft or multi) else 0,
            stall_tolerance=PUSH_SOFT_STALL_TOLERANCE if soft else None,
            # The soft cube's compliant contact (PUSH_SOFT_PENALTY) overlaps more than the
            # rigid limit; its penetration is reported, not bounded.
            max_penetration=None if soft else MAX_PENETRATION,
            step_losses=step_losses,
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
            shape=cube_shape(GRASP_CUBE_HALF),
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


def spawn_hand_arm(profile: HandGrasp, scene, with_controller: bool = True, with_contact: bool = True):
    """FR3 arm with the five-finger hand of the profile; the finger joints get the gripper
    gains, the arm the reach/push gains times GRASP_ARM_GAIN_SCALE. The returned
    end-effector actor is the profile's wrist link."""
    arm_gains = _arm_gains()

    def gains_of(name):
        if name.startswith("fr3"):
            k, d = arm_gains(name)
            return GRASP_ARM_GAIN_SCALE * k, GRASP_ARM_GAIN_SCALE * d
        return GRIPPER_GAINS

    return spawn_bot(
        scene, profile.bot, profile.wrist_link, gains_of, with_controller, with_contact, GRASP_CONTACT
    )


def revolute_dof_indices(bot) -> dict[str, int]:
    """Index in the articulated pose vector of every revolute joint of a fixed-base bot,
    by joint name (the welded joints carry no degree of freedom)."""
    indices: dict[str, int] = {}
    joints = bot.get_bot_prefab().joints
    for i in range(len(joints)):
        joint = joints[i]
        if joint.type == physics.ArticulatedJointType.REVOLUTE:
            indices[joint.name] = len(indices)
    if len(indices) != bot.get_articulated_actor().get_num_dofs():
        raise RuntimeError("the bot has degrees of freedom that are not revolute joints")
    return indices


def hand_ik_poses(points, spawner, link_name: str, rotation: np.ndarray, flange_start: float):
    """Joint poses placing the origin of the link whose name ends with ``link_name`` at
    each point with the world orientation ``rotation`` (a 3x3 matrix whose columns are
    the link's axes in the world), from a dedicated zero-gravity IK scene. The engine's IK
    solver takes one position and one rotation target per link and solves from the
    current pose; ``flange_start`` is the arm's joint 7 at the start (the pose it converges
    to depends on that side of the joint's range). A pose that is not reached within a
    millimetre and half a degree is an error."""
    scene = physics.create_scene("ik")
    scene.set_gravity([0.0, 0.0, 0.0])
    bot, arm, link, context = spawner(scene, with_controller=False, with_contact=False)
    if link_name is not None and not link.get_name().endswith(link_name):
        links = []
        scene.for_each_actor(lambda a: links.append(a) if a.is_nested_link_actor() else None)
        link = [a for a in links if a.get_name().endswith(link_name)][0]
    solver = physics.experimental.create_ik_solver(scene)
    params = solver.get_solver_params()
    params.max_iter = 2000
    params.abs_tol = 1e-8
    params.rel_tol = 1e-10
    params.position_error_thres = 1e-4
    params.rotation_error_thres = 1e-4
    solver.set_solver_params(params)
    rotation = np.asarray(rotation, dtype=np.float64)
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12) or np.linalg.det(rotation) < 0:
        raise ValueError("rotation must be a proper rotation matrix")
    q_start = np.zeros(arm.get_num_dofs())
    arm.get_articulated_pose(q_start)
    q_start[6] = flange_start
    arm.set_articulated_pose_from_joints(q_start)
    poses = []
    for point in points:
        point = np.asarray(point, dtype=np.float64)
        solver.create_position_target(link.get_handle(), [0.0, 0.0, 0.0], list(point), 1.0)
        solver.create_rotation_target(
            link.get_handle(), [0.0, 0.0, 0.0], list(_rotation_vector(rotation)), 1.0
        )
        converged = solver.solve_ik()
        q = np.zeros(arm.get_num_dofs())
        arm.get_articulated_pose(q)
        transform = link.get_root_transform()
        reached = np.asarray(transform.translation, dtype=np.float64)
        reached_rotation = _quaternion_matrix(np.asarray(transform.rotation.tolist(), dtype=np.float64))
        cosine = np.clip((np.trace(rotation.T @ reached_rotation) - 1.0) / 2.0, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine))
        distance = np.linalg.norm(reached - point)
        print(
            f"  IK {link_name} {np.round(point, 3)} -> {reached.round(4)} "
            f"({distance * 1000:.2f} mm, {angle:.2f} deg off, joint 7 at {q[6]:.2f})"
        )
        if not converged or distance > 1e-3 or angle > 0.5:
            raise RuntimeError(
                f"IK did not reach the pose at {np.round(point, 3)}: {distance * 1000:.1f} mm and "
                f"{angle:.1f} deg off (converged: {converged})"
            )
        poses.append(q)
    solver.clear_position_target(link.get_handle())
    solver.clear_rotation_target(link.get_handle())
    robotics.destroy_bot(scene, bot)
    physics.experimental.destroy_ik_solver(solver)
    return poses


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Axis times angle of a rotation matrix."""
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle < 1e-12:
        return np.zeros(3)
    if angle > np.pi - 1e-6:
        # Rotation by pi: the axis is the eigenvector for eigenvalue 1.
        values, vectors = np.linalg.eigh(rotation + rotation.T)
        axis = vectors[:, np.argmax(values)]
        return np.pi * axis / np.linalg.norm(axis)
    axis = np.array(
        [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]]
    ) / (2.0 * np.sin(angle))
    return angle * axis


def _quaternion_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of a unit quaternion (x, y, z, w)."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def build_hand_grasp_task(profile: HandGrasp):
    """Scene, actors and initial controls of a five-finger grasp (see :class:`HandGrasp` and
    the module docstring); the same return value as :func:`build_grasp_task`."""
    name = f"robot_{profile.name}_grasp"
    print(f"[{name}] IK for the descend / grasp / lift wrist poses")
    grasp = HAND_CUBE_POS + np.array(profile.offset)
    spawner = functools.partial(spawn_hand_arm, profile)
    q_pre, q_grasp, q_lift = hand_ik_poses(
        [grasp + np.array([0.0, 0.0, 0.15]), grasp, grasp + np.array([0.0, 0.0, 0.25])],
        spawner,
        profile.wrist_link,
        profile.rotation,
        profile.flange_start,
    )
    scene = physics.create_scene(name)
    scene.set_gravity([0.0, 0.0, -9.81])
    bot, arm, wrist, context = spawner(scene)
    scene.create_rigid_actor(
        name="ground",
        shape=physics.create_plane_shape(normal=[0.0, 0.0, 1.0], distance=0.0),
        is_static=True,
        contact=GRASP_CONTACT,
    )
    cube = scene.create_rigid_actor(
        name="cube",
        shape=cube_shape(HAND_CUBE_HALF),
        density=HAND_CUBE_DENSITY,
        contact=GRASP_CONTACT,
        world_from_local=physics.TransformRT(HAND_CUBE_POS.tolist()),
    )
    dofs = revolute_dof_indices(bot)
    unknown = (set(profile.fingers) | set(profile.thumb) | set(profile.thumb_pre or {}) | set(profile.open)) - set(dofs)
    if unknown:
        raise ValueError(f"the profile {profile.name} names joints the hand does not have: {sorted(unknown)}")
    n = arm.get_num_dofs()
    q0 = np.zeros(n)
    arm.get_articulated_pose(q0)

    def hand_targets(values: dict[str, float], base: np.ndarray) -> np.ndarray:
        h = base.copy()
        for joint, value in values.items():
            h[dofs[joint] - 7] = value
        return h

    hand_open = hand_targets(profile.open, q0[7:])
    for q in (q_pre, q_grasp, q_lift):
        q[7:] = hand_open  # the IK moves the fingers too; keep the hand open
    arm.set_articulated_pose_from_joints(q_pre)
    configure(scene)
    fingers_closed = hand_targets(profile.fingers, hand_open)
    thumb = [dofs[joint] - 7 for joint in profile.thumb]
    thumb_open = hand_open[thumb].copy()
    thumb_closed = np.array([profile.thumb[joint] for joint in profile.thumb])
    thumb_pre = None if profile.thumb_pre is None else np.array([profile.thumb_pre[joint] for joint in profile.thumb])

    def hand_closed(fraction: float) -> np.ndarray:
        h = hand_open + fraction * (fingers_closed - hand_open)
        if thumb_pre is None:
            h[thumb] = thumb_open + fraction * (thumb_closed - thumb_open)
        elif fraction < profile.thumb_swing:
            h[thumb] = thumb_open + (fraction / profile.thumb_swing) * (thumb_pre - thumb_open)
        else:
            a = (fraction - profile.thumb_swing) / (1.0 - profile.thumb_swing)
            h[thumb] = thumb_pre + a * (thumb_closed - thumb_pre)
        return h

    num_steps = HAND_LIFT_END
    dt = 0.02
    controls0 = interpolate_targets(
        [(0, q_pre), (HAND_CLOSE_START, q_grasp), (HAND_CLOSE_START + HAND_CLOSE_STEPS, q_grasp), (num_steps, q_lift)],
        num_steps,
    )
    for step in range(num_steps):
        close = 0.0 if step < HAND_CLOSE_START else min(1.0, (step - HAND_CLOSE_START) / HAND_CLOSE_STEPS)
        controls0[step, 7:] = hand_closed(close)

    def cube_position() -> np.ndarray:
        return np.asarray(cube.get_center_of_mass_transform().translation)

    # Where the cube ends up in the grasp after the lift (measured on the initial trajectory),
    # displaced sideways: the optimizer has to carry it there.
    goal = np.array(profile.lifted_cube) + GRASP_GOAL_OFFSET
    return scene, bot, arm, cube, cube_position, context, controls0, goal, dt


def task_grasp(output_dir: pathlib.Path, num_iterations: int, check: bool) -> None:
    _task_grasp_impl(output_dir, num_iterations, check, soft=False)


def task_hand_grasp(output_dir: pathlib.Path, num_iterations: int, check: bool, hand: str = "hand") -> None:
    """A five-finger grasp (``hand`` names a profile of HAND_GRASPS; see
    :func:`build_hand_grasp_task`), optimized like :func:`task_grasp`."""
    _task_grasp_impl(output_dir, num_iterations, check, soft=False, hand=hand)


def task_grasp_soft(output_dir: pathlib.Path, num_iterations: int, check: bool) -> None:
    """The grasp with a soft cube (see :func:`task_grasp`); failure-adaptive substeps as in
    the soft push."""
    _task_grasp_impl(output_dir, num_iterations, check, soft=True)


def _task_grasp_impl(
    output_dir: pathlib.Path, num_iterations: int, check: bool, soft: bool, hand: str = ""
) -> None:
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
    if soft and hand:
        raise ValueError("soft and hand are exclusive")
    if hand:
        if hand not in HAND_GRASPS:
            raise ValueError(f"unknown hand {hand!r}; the profiles are {sorted(HAND_GRASPS)}")
        profile = HAND_GRASPS[hand]
        name, carry_start = f"robot_{profile.name}_grasp", HAND_CARRY_START
        title_hand = f"{profile.title}: normalized gradient descent on the carry knots"
        max_penetration, stall_tolerance = profile.max_penetration, profile.stall_tolerance
        scene, bot, arm, cube, cube_position, context, controls0, goal, dt = build_hand_grasp_task(profile)
    else:
        name = "robot_grasp_soft" if soft else "robot_grasp"
        carry_start = GRASP_CARRY_START
        title_hand = None
        # The soft cube's 1e7 contact overlaps more than the rigid limit (reported, not bounded).
        max_penetration = None if soft else MAX_PENETRATION
        stall_tolerance = PUSH_SOFT_STALL_TOLERANCE if soft else None
        scene, bot, arm, cube, cube_position, context, controls0, goal, dt = build_grasp_task(
            soft, fingertip_box_dir=output_dir / "fingertip_boxes" if soft else None
        )
    # Optimize the arm's targets of the carry phase; the fingers keep their closure
    # (a finger shoving the cube sideways is the quickest way to move it, and to drop it).
    trainable = np.zeros(controls0.shape, dtype=bool)
    trainable[carry_start:, :7] = True
    # The carry-phase arm targets are the linear interpolation of GRASP_NUM_KNOTS knots.
    # The first knot is pinned to the trajectory at the carry start, so an update is a
    # ramp starting from rest rather than a jump (a jerk at the carry start drops the
    # cube, as does moving every per-step target by the learning rate at once); the
    # other knots are the optimization variables. The initial carry phase is linear, so
    # the knots reproduce it exactly.
    num_steps = controls0.shape[0]
    knots = np.linspace(carry_start, num_steps - 1, GRASP_NUM_KNOTS).round().astype(int)
    steps = np.arange(carry_start, num_steps)
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
        arm_targets = torch.cat([base[:carry_start, :7], carry], dim=0)
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
                diffsim.get_center_of_mass_transform_backward(cube, g)

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
                title_hand
                if hand
                else (
                    "FR3 + 2F-85 grasp of a soft cube: normalized gradient descent on the carry knots"
                    if soft
                    else "FR3 + 2F-85 grasp: normalized gradient descent on the carry knots through frictional contact"
                )
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
            substep_levels=PUSH_SUBSTEP_LEVELS if (soft or hand) else 0,
            # The hands' pinches squeeze the fingertips into the cube by design.
            max_penetration=max_penetration,
            stall_tolerance=stall_tolerance,
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
        shape=cube_shape(HAUL_BOX_HALF),
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
            "reach", "push", "push_soft", "push_multi", "push_policy", "grasp", "grasp_soft",
            *HAND_GRASPS, "tendon", "haul", "both", "all",
        ],
        default="all",
    )
    parser.add_argument("--check", action="store_true", help="finite-difference gradient check first")
    parser.add_argument(
        "--export-scenes",
        action="store_true",
        help="also export the recorded frames for render_diffsim_blender.py",
    )
    args = parser.parse_args()
    example_diffsim_video.EXPORT_SCENES = args.export_scenes
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
    if args.task in ("push_policy", "all"):
        task_push_policy(args.output_dir, args.iterations, args.check)
    if args.task in ("grasp", "all"):
        task_grasp(args.output_dir, args.iterations, args.check)
    if args.task in ("grasp_soft", "all"):
        task_grasp_soft(args.output_dir, args.iterations, args.check)
    for hand in HAND_GRASPS:
        if args.task in (hand, "all"):
            task_hand_grasp(args.output_dir, args.iterations, args.check, hand)
    if args.task in ("tendon", "all"):
        task_tendon(args.output_dir, args.iterations, args.check)
    if args.task in ("haul", "all"):
        task_haul(args.output_dir, args.iterations, args.check)
    print(f"done in {time.time() - start:.1f} s")
    physics.shutdown()


if __name__ == "__main__":
    main()
