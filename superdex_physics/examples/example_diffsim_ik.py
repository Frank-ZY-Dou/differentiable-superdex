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
"""Example: inverse kinematics through the differentiable simulator.

The unknowns are the joint targets of the FR3 arm's pose controller. A short rollout lets
the controller settle the arm on them, under gravity; the loss is the end effector's
position and orientation error at the end of the rollout; its gradient with respect to
the targets comes from the adjoint through the rollout
(``superdex.physics.diffsim_torch.TorchRollout``); Adam updates the targets. Two problems:

1. Free space. Reach a target position with the tool orientation of the home pose. The
   engine's quasi-static IK solver gives the kinematic solution of the position target;
   used as controller targets, the arm settles a few millimetres below it, because the
   controller's finite stiffness yields to gravity. The differentiable IK optimizes the
   settled pose, so it compensates that sag, and it constrains the orientation as well.
2. Through contact, with a force objective. The wrist has to rest on the top of a box at a
   given point, with the home orientation, pressing with 10 N: the loss is the wrist's
   horizontal position and orientation error plus the contact-force error, whose gradient
   is the engine's contact-force adjoint, and the height is whatever gives 10 N. Kinematic
   IK has no notion of this; the solution's interpenetration is reported with
   ``PenetrationChecker`` (a fraction of a millimetre at 10 N).

Before the first update the adjoint gradient is checked against central finite
differences of the rollout loss along the gradient direction. Requires double precision
(selected below, before the first physics import), the robotics assets and PyTorch. No GUI.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("SUPERDEX_PRECISION", "double")

import numpy as np
import superdex.physics as physics
import superdex.robotics as robotics
import torch
from superdex.physics import diffsim
from superdex.physics.diffsim_torch import TorchRollout
from superdex.physics.utils.penetration import PenetrationChecker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from example_diffsim_robot_video import (  # noqa: E402
    ARM_CONTACT,
    GRAVITY,
    configure,
    ik_joint_poses,
    spawn_arm,
)
from example_diffsim_video import box_tet_mesh  # noqa: E402

DT = 0.01
NUM_STEPS = 60  # 0.6 s: the controller settles the arm on constant targets
ORIENTATION_WEIGHT = 0.04  # 0.1 rad of orientation error weighs like 1 cm of position error


class PoseLoss:
    """0.5 |p - p*|^2 + w 0.5 |q - q*|^2 on a link's frame (its root transform: the point the
    engine's IK solver targets too; quaternion XYZW, the sign of q* matched to q), with the
    engine's transform adjoint."""

    def __init__(self, link, position, quaternion, weight: float):
        self.link = link
        self.position = np.asarray(position, dtype=np.float64)
        self.quaternion = np.asarray(quaternion, dtype=np.float64)
        self.weight = weight
        self.last = None  # the errors of the last value(): the rollout's final state

    def _errors(self):
        transform = self.link.get_root_transform()
        p = np.asarray(transform.translation, dtype=np.float64)
        q = np.asarray(transform.rotation, dtype=np.float64)
        target = self.quaternion if q @ self.quaternion >= 0.0 else -self.quaternion
        return p - self.position, q - target

    def value(self) -> float:
        dp, dq = self._errors()
        self.last = (dp, dq)
        return 0.5 * float(dp @ dp) + self.weight * 0.5 * float(dq @ dq)

    def accumulate_output_grad(self) -> None:
        dp, dq = self._errors()
        grad = np.concatenate([dp, self.weight * dq])
        diffsim.get_root_transform_backward(self.link, grad)

    def report(self) -> str:
        """The errors at the end of the last rollout (the scene itself is restored to the
        initial state once a rollout's adjoint sweep is done)."""
        dp, dq = self.last
        angle = 2.0 * np.degrees(np.arcsin(min(1.0, np.linalg.norm(dq) / 2.0)))
        return f"position error {1000 * np.linalg.norm(dp):.2f} mm, orientation error {angle:.2f} deg"


class PressLoss(PoseLoss):
    """The pose loss on the horizontal position and the orientation only, plus
    0.5 w (F_z - F*)^2 on the vertical contact force the link receives (the engine's
    contact-force query and its adjoint)."""

    def __init__(self, link, position, quaternion, weight: float, force: float, force_weight: float):
        super().__init__(link, position, quaternion, weight)
        self.force = force
        self.force_weight = force_weight

    def _errors(self):
        dp, dq = super()._errors()
        dp = dp.copy()
        dp[2] = 0.0
        return dp, dq

    def _force_error(self) -> float:
        return float(np.asarray(self.link.get_contact_force_world(), dtype=np.float64)[2]) - self.force

    def value(self) -> float:
        df = self._force_error()
        self.last_force = df + self.force
        return super().value() + self.force_weight * 0.5 * df * df

    def accumulate_output_grad(self) -> None:
        super().accumulate_output_grad()
        df = self._force_error()
        diffsim.get_contact_force_world_backward(self.link, np.array([0.0, 0.0, self.force_weight * df]))

    def report(self) -> str:
        return f"{super().report()}, vertical contact force {self.last_force:.2f} N (target {self.force:.0f} N)"


def settle(bridge, targets: np.ndarray) -> float:
    """The loss of constant joint targets held for the whole rollout."""
    controls = torch.tensor(np.tile(targets, (NUM_STEPS, 1)), dtype=torch.float64)
    return float(bridge(controls=controls))


def check_gradient(bridge, targets: np.ndarray, gradient: np.ndarray, steps) -> None:
    """Central differences of the loss along the gradient direction against the adjoint."""
    direction = gradient / np.linalg.norm(gradient)
    predicted = float(gradient @ direction)
    quotients = []
    for h in steps:
        plus = settle(bridge, targets + h * direction)
        minus = settle(bridge, targets - h * direction)
        quotients.append((plus - minus) / (2.0 * h))
    worst = max(abs(q - predicted) / abs(predicted) for q in quotients)
    print(
        f"  gradient check: adjoint {predicted:.6e}, finite differences "
        + ", ".join(f"{q:.6e} (h={h:g})" for q, h in zip(quotients, steps))
        + f"; relative error {worst:.1e}"
    )


def solve(bridge, loss: PoseLoss, targets0: np.ndarray, iterations: int, fd_steps):
    """L-BFGS with a strong-Wolfe line search on the constant joint targets (every function
    evaluation is a rollout with its adjoint). Returns the best targets and the loss
    history of the evaluations."""
    q = torch.tensor(targets0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [q], lr=1.0, max_iter=iterations, history_size=20, line_search_fn="strong_wolfe",
        tolerance_grad=1e-12, tolerance_change=1e-14,
    )
    history = []
    best = [np.inf, targets0.copy()]

    def closure():
        optimizer.zero_grad()
        value = bridge(controls=q.unsqueeze(0).expand(NUM_STEPS, -1))
        value.backward()
        history.append(float(value))
        if float(value) < best[0]:
            best[0], best[1] = float(value), q.detach().numpy().copy()
        if len(history) == 1:
            check_gradient(bridge, q.detach().numpy(), q.grad.numpy().copy(), fd_steps)
        if len(history) % 10 == 1:
            print(f"  evaluation {len(history):3d}: loss {float(value):.3e}, {loss.report()}")
        return value

    optimizer.step(closure)
    settle(bridge, best[1])
    return best[1], history


def free_space(iterations: int) -> None:
    print("Problem 1: free space, position and orientation")
    scene = physics.create_scene("differentiable IK")
    scene.set_gravity(GRAVITY)
    try:
        bot, arm, ee, context = spawn_arm(scene, with_controller=True, with_contact=False)
        configure(scene)
        home = np.zeros(arm.get_num_dofs())
        arm.get_articulated_pose(home)
        arm.set_articulated_target_pose(home)
        arm.set_articulated_target_velocity(np.zeros_like(home))
        start = ee.get_root_transform()
        position = np.asarray(start.translation, dtype=np.float64) + np.array([0.12, 0.2, -0.18])
        quaternion = np.asarray(start.rotation, dtype=np.float64)
        loss = PoseLoss(ee, position, quaternion, ORIENTATION_WEIGHT)
        bridge = TorchRollout(scene, dt=DT, num_steps=NUM_STEPS, control_actors=[arm], terminal_losses=[loss])
        try:
            print(f"  target {np.round(position, 3)}, the home orientation")
            settle(bridge, home)
            print(f"  home targets held: {loss.report()}")
            kinematic = ik_joint_poses([position])[0]
            settle(bridge, kinematic)
            print(f"  kinematic IK targets held (position target only): {loss.report()}")
            targets, history = solve(bridge, loss, kinematic, iterations, fd_steps=(1e-5, 1e-6))
            print(f"  differentiable IK: {loss.report()} after {len(history)} evaluations, loss {history[0]:.3e} -> {min(history):.3e}")
        finally:
            bridge.close()
        robotics.destroy_bot(scene, bot)
    finally:
        physics.destroy_scene(scene)


def through_contact(iterations: int) -> None:
    print("Problem 2: through contact, resting the wrist on a box with a 10 N force objective")
    scene = physics.create_scene("differentiable IK through contact")
    scene.set_gravity(GRAVITY)
    try:
        bot, arm, ee, context = spawn_arm(scene, with_controller=True, with_contact=True)
        home = np.zeros(arm.get_num_dofs())
        arm.get_articulated_pose(home)
        arm.set_articulated_target_pose(home)
        arm.set_articulated_target_velocity(np.zeros_like(home))
        # The flange link carries no contact samples; the wrist link (fr3_link7) is what
        # touches the box, so it is the link positioned and the link whose force is read.
        links = []
        scene.for_each_actor(lambda actor: links.append(actor) if actor.is_nested_link_actor() else None)
        wrist = [actor for actor in links if actor.get_name().endswith("fr3_link7")][0]
        start = wrist.get_root_transform()
        start_position = np.asarray(start.translation, dtype=np.float64)
        top = start_position[2] - 0.18
        center = start_position[:2] + np.array([0.1, 0.15])
        size = 0.3
        coordinates, connectivity = box_tet_mesh(size, 2)
        scene.create_rigid_actor(
            name="box",
            shape=physics.create_tet_mesh_shape(coordinates=coordinates, connectivity=connectivity),
            is_static=True,
            contact=ARM_CONTACT,
            world_from_local=physics.TransformRT([center[0], center[1], top - size / 2.0]),
        )
        configure(scene)
        wrist.register_query(physics.QueryType.TOTAL_CONTACT_FORCE)
        checker = PenetrationChecker(scene)
        initial = scene.capture_state()
        quaternion = np.asarray(start.rotation, dtype=np.float64)
        loss = PressLoss(
            wrist, np.array([center[0], center[1], 0.0]), quaternion, ORIENTATION_WEIGHT,
            force=10.0, force_weight=1e-6,  # 1 N of force error weighs like 1 mm of position error
        )
        bridge = TorchRollout(scene, dt=DT, num_steps=NUM_STEPS, control_actors=[arm], terminal_losses=[loss])
        try:
            print(f"  box top at z = {top:.3f} m, wrist target above {np.round(center, 3)}, home orientation, 10 N")
            # Start in contact: the kinematic solution of a wrist point 5 mm inside the box top
            # (the force gradient is zero while the wrist is in the air).
            kinematic = ik_joint_poses([np.array([center[0], center[1], top - 0.005])], link_name="fr3_link7")[0]
            settle(bridge, kinematic)
            print(f"  kinematic IK targets held: {loss.report()}")
            targets, history = solve(bridge, loss, kinematic, iterations, fd_steps=(1e-6, 1e-7))
            print(
                f"  differentiable IK: {loss.report()} after {len(history)} evaluations, loss "
                f"{history[0]:.3e} -> {min(history):.3e}"
            )
        finally:
            bridge.close()
        # Replay the solution with the interpenetration checker: penalty contact overlaps under
        # load by an amount the report shows (see PenetrationChecker).
        # The transient (the wrist landing on the box) and the settled end are reported apart.
        settled = PenetrationChecker(scene)
        scene.restore_state(initial, False)
        arm.set_articulated_target_pose(np.ascontiguousarray(targets))
        for step in range(NUM_STEPS):
            scene.step(DT)
            checker.record(step)
            if step >= NUM_STEPS - 10:
                settled.record(step)
        force = np.asarray(wrist.get_contact_force_world(), dtype=np.float64)
        print(f"  contact force on the wrist at the end of the replay: {np.round(force, 2)} N")
        print("  interpenetration while the wrist lands on the box, then settled:")
        for report in (checker.report(), settled.report()):
            for line in report.splitlines()[1:]:
                print(f"  {line}")
        scene.release_state(initial)
        robotics.destroy_bot(scene, bot)
    finally:
        physics.destroy_scene(scene)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--problem", choices=["free", "contact", "both"], default="both")
    parser.add_argument("--iterations", type=int, default=60)
    args = parser.parse_args()
    physics.initialize(num_worker_threads=0)
    start = time.time()
    if args.problem in ("free", "both"):
        free_space(args.iterations)
    if args.problem in ("contact", "both"):
        through_contact(args.iterations)
    print(f"done in {time.time() - start:.1f} s")
    physics.shutdown()


if __name__ == "__main__":
    main()
