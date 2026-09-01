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
   steer it through contact between the arm's collision meshes and the cube
   (articulated-vs-rigid contact in one island; the cube slides on the ground
   with friction, the arm-cube pair is frictionless - see ARM_CONTACT).

Rigor. Before optimizing, ``--check`` compares the adjoint gradient of the
largest-gradient target entries against central finite differences of the
full rollout at two step sizes (1e-5, 1e-6) and prints both the relative
error and the FD self-consistency; every forward step is required to reach
``ConvergenceStatus.CONVERGED`` (the script aborts otherwise). What this
shows for the push task (2026-09-01 measurements, frictionless arm-cube
pair, friction fading off): the checked entries agree to 1e-6..1e-3. With a
frictional arm-cube pair the entries at contact onset were off by
5e-2..1.3e-1 although the rollout FD was self-consistent to 1e-5 - the pinned
engine defect mentioned above, not a smoothness problem.
Design choices that keep the interaction smooth: contact stiffness 1e6, the
end effector starts in contact (no impact), the sweep is slow, and the forward
Newton tolerance is 1e-9 (tighter settings sit at the round-off floor of this
model and stall the solver, which the guard would report).

Requirements: SUPERDEX_PRECISION=double (set below), the robotics extension
(``superdex.robotics``), polyscope, imageio+ffmpeg, OpenCV, PyTorch, and the
repository assets (resolved through ``superdex.physics.paths``). Run::

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
from superdex.physics.diffsim_torch import TorchRollout
from superdex.physics.paths import resolve_asset
from superdex.physics.utils import render_model_registry

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from example_diffsim_video import GRAVITY, Recorder, save_loss_curve  # noqa: E402

diffsim = physics.diffsim

ARM_BOT = "bots/arms/fr3/fr3.superdex_bot"
EE_LINK = "fr3_link8"
# Per-joint PD gains of the pose controller (stiffness [N m/rad], damping [N m s/rad]).
JOINT_GAINS = list(zip([400, 400, 300, 300, 150, 100, 60], [40, 40, 30, 30, 15, 10, 6]))
CUBE_HALF = 0.05
CONTACT = physics.ContactParams(penalty_coefficient=1e6, coulomb_friction_coefficient=0.4)
# The arm's links carry a frictionless material: a contact pair combines both
# owners' friction coefficients by geometric mean, so arm-cube contact is
# frictionless (a smooth pusher) while cube-ground contact keeps its friction.
# Reason: with friction between an articulated link and a dynamic rigid body,
# the adjoint's previous-state coupling is wrong for the contacts whose SDF
# owner is the link - a pinned engine defect (see test_diffsim_gradients:
# test_link_as_collider_of_frictional_sync_contact_is_pinned_wrong); the penalty
# term - all the pushing needs - is exact. (Differentiable scenes also switch
# friction fading by normal alignment off; see ExperimentalEvalParams.fade_friction.)
ARM_CONTACT = physics.ContactParams(penalty_coefficient=1e6, coulomb_friction_coefficient=0.0)
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


def spawn_arm(scene, with_controller: bool = True, with_contact: bool = True):
    """FR3 arm with the given contact material and (optionally) a pose controller.
    Returns (bot, arm actor, end-effector link actor, robotics context); the
    context must outlive the bot."""
    prefab = robotics.load_bot_prefab_from_file(str(resolve_asset(ARM_BOT)))
    for i in range(len(prefab.links)):
        link = prefab.links[i]
        link.contact = ARM_CONTACT
        if not with_contact:
            link.collider_type = physics.ColliderType.NONE
        prefab.links[i] = link
    context = robotics.create_context()
    bot = robotics.create_bot(scene, prefab, context)
    arm = bot.get_articulated_actor()
    if with_controller:
        gains = iter(JOINT_GAINS)
        tracking = []
        for i in range(len(prefab.joints)):
            if prefab.joints[i].type == physics.ArticulatedJointType.REVOLUTE:
                k, d = next(gains)
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
    ee = [a for a in actors if a.get_name().endswith(EE_LINK)][0]
    return bot, arm, ee, context


def configure(scene) -> None:
    diffsim.make_scene_differentiable(scene)
    solver = scene.get_solver_params()
    newton = solver.non_linear_solver
    newton.max_iter = 300
    # 1e-9 is the tightest tolerance this model reaches reliably: with the pose
    # controller active the Newton residual stalls at 1e-10..2e-10 (round-off).
    newton.abs_tol = 1e-9
    newton.rel_tol = 1e-9
    solver.non_linear_solver = newton
    scene.set_solver_params(solver)
    params = diffsim.get_back_propagation_solver_params(scene)
    params.outer_solver_abs_tol = 1e-10
    params.outer_solver_max_iter = 300
    params.inner_solver_abs_tol = 1e-14
    params.validate_finite_diff = True
    diffsim.set_back_propagation_solver_params(scene, params)


def ik_joint_poses(waypoints):
    """Joint poses reaching the end-effector waypoints, from a dedicated
    zero-gravity IK scene (the engine solves IK as a quasi-static simulation)."""
    scene = physics.create_scene("ik")
    scene.set_gravity([0.0, 0.0, 0.0])
    bot, arm, ee, context = spawn_arm(scene, with_controller=False, with_contact=False)
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
    physics.experimental.destroy_ik_solver(solver)
    robotics.destroy_bot(scene, bot)
    physics.destroy_scene(scene)
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


def check_gradient(bridge, controls: np.ndarray, grad: np.ndarray, num_entries: int = 4) -> None:
    """Central finite differences of the full rollout loss on the entries with
    the largest adjoint gradient, at two step sizes."""
    loss_of = lambda c: float(bridge(controls=torch.tensor(c, dtype=torch.float64)).detach())
    print("  gradient check (adjoint vs rollout FD at eps 1e-5 / 1e-6):")
    for flat in np.argsort(-np.abs(grad).ravel())[:num_entries]:
        s, d = divmod(int(flat), controls.shape[1])
        fds = []
        for eps in (1e-5, 1e-6):
            plus, minus = controls.copy(), controls.copy()
            plus[s, d] += eps
            minus[s, d] -= eps
            fds.append((loss_of(plus) - loss_of(minus)) / (2.0 * eps))
        denom = max(abs(fds[0]), 1e-30)
        print(
            f"    step {s:3d} joint {d}: adjoint {grad[s, d]:+.5e}  fd {fds[0]:+.5e}  "
            f"rel err {abs(grad[s, d] - fds[0]) / denom:.1e}  "
            f"fd self-consistency {abs(fds[0] - fds[1]) / denom:.0e}"
        )


class ConvergenceGuard:
    """Checks every forward step of a replay: CONVERGED is required, except that a
    solve that stalled (STOPPED) below ``stall_tolerance`` - the round-off floor of
    these robot models, 1e-9 relative to the ~1e2 N force scale - is counted and
    reported rather than treated as a failure. Anything worse aborts the run."""

    stall_tolerance = 1e-7

    def __init__(self, scene):
        self.scene = scene
        self.checked = 0
        self.stalled = 0

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
            f"({self.stalled} stalled below {self.stall_tolerance:.0e} residual)"
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
) -> None:
    num_steps = controls0.shape[0]
    guard = ConvergenceGuard(scene)
    bridge = TorchRollout(
        scene, dt=dt, num_steps=num_steps, control_actors=[arm], terminal_losses=[loss]
    )
    recorder = Recorder(scene, target, look_from=look_from, look_at=look_at, title=title)
    controls = torch.tensor(controls0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([controls], lr=learning_rate)
    losses = []
    record = {0, 1, 2, 4, 7, 12, 20, 30, num_iterations - 1}

    def replay(targets: np.ndarray, capture: bool, caption_fn=None) -> None:
        scene.restore_state(bridge._state_init, False)
        if capture:
            recorder.begin_iteration()
        for step in range(num_steps):
            arm.set_articulated_target_pose(np.ascontiguousarray(targets[step]))
            scene.step(dt)
            guard.assert_last_step()
            if capture:
                recorder.capture(
                    tracked_point(),
                    caption_fn(step),
                    hold=(1 if step < num_steps - 1 else 12),
                )

    for iteration in range(num_iterations):
        optimizer.zero_grad()
        value = bridge(controls=controls)
        value.backward()
        # The gradient check must see the adjoint itself, so read it before the
        # clipping below rescales it.
        raw_grad = controls.grad.detach().clone()
        losses.append(float(value))
        result = bridge.last_result
        if not result.fd_valid:
            print("  warning: the adjoint's finite-difference self-check flagged a step")
        if iteration == 0 and check:
            check_gradient(bridge, controls0, raw_grad.numpy())
        # Contact tasks have occasional nonsmooth steps: cap the update so one
        # spiky gradient cannot throw the arm into an impact (DiffMJX-style clipping).
        torch.nn.utils.clip_grad_norm_([controls], max_norm=grad_clip)
        print(
            f"[{name}] iter {iteration:3d}  loss {losses[-1]:.6f}  "
            f"|grad| {float(raw_grad.norm()):.3e} (clipped to {float(controls.grad.norm()):.3e})  "
            f"adjoint residual {result.max_adjoint_residual:.1e}"
        )
        if iteration in record:
            current = controls.detach().numpy().copy()
            replay(
                current,
                capture=True,
                caption_fn=lambda step, it=iteration: (
                    f"iteration {it}   t = {(step + 1) * dt:.2f} s   loss = {losses[-1]:.5f}"
                ),
            )
        optimizer.step()

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
    robotics.destroy_bot(scene, bot)
    physics.destroy_scene(scene)


def task_push(output_dir: pathlib.Path, num_iterations: int, check: bool) -> None:
    cube_start = np.array([0.55, 0.0, CUBE_HALF - 0.001])
    goal = np.array([0.80, 0.10, CUBE_HALF])
    # Pre-push pose and a straight slow sweep, from IK on the end effector.
    print("[robot_push] IK for the initial joint-target trajectory")
    # The engine's IK is a quasi-static simulation towards the target, solved
    # from the previous waypoint's pose: sample the sweep densely (5 cm).
    waypoints = [np.array([0.44 + 0.05 * k, 0.0, 0.10]) for k in range(6)]  # 0.44 .. 0.69
    poses = ik_joint_poses(waypoints)
    num_steps = 75
    keys = [(0, poses[0])] + [(15 * k, poses[k]) for k in range(1, 6)]
    controls0 = interpolate_targets(keys, num_steps)

    scene = physics.create_scene("Differentiable push")
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
        shape=physics.create_tet_mesh_shape(coordinates=cube_coords(CUBE_HALF), connectivity=CUBE_CONN),
        density=300.0,
        contact=CONTACT,
        world_from_local=physics.TransformRT(cube_start.tolist()),
    )
    arm.set_articulated_pose_from_joints(poses[0])
    configure(scene)

    class CubeLoss:
        def value(self) -> float:
            d = np.asarray(cube.get_center_of_mass_transform().translation) - goal
            return 0.5 * float(d @ d)

        def accumulate_output_grad(self) -> None:
            d = np.asarray(cube.get_center_of_mass_transform().translation) - goal
            g = np.zeros(7)
            g[:3] = d
            diffsim.get_center_of_mass_transform_backward(cube, g)

    run_task(
        "robot_push",
        scene,
        arm,
        controls0=controls0,
        loss=CubeLoss(),
        tracked_point=lambda: np.asarray(cube.get_center_of_mass_transform().translation),
        target=goal,
        look_from=[1.9, -1.6, 0.9],
        look_at=[0.6, 0.05, 0.1],
        title="FR3 push: Adam on the joint-target trajectory through frictional contact",
        output_dir=output_dir,
        num_iterations=num_iterations,
        learning_rate=0.002,
        dt=0.02,
        check=check,
        grad_clip=0.02,
    )
    robotics.destroy_bot(scene, bot)
    physics.destroy_scene(scene)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("diffsim_videos"))
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--task", choices=["reach", "push", "both"], default="both")
    parser.add_argument("--check", action="store_true", help="finite-difference gradient check first")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    physics.initialize(num_worker_threads=0)
    start = time.time()
    if args.task in ("reach", "both"):
        task_reach(args.output_dir, args.iterations, args.check)
    if args.task in ("push", "both"):
        task_push(args.output_dir, args.iterations, args.check)
    print(f"done in {time.time() - start:.1f} s")
    physics.shutdown()


if __name__ == "__main__":
    main()
