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

"""Gradient-consistency harness for ``superdex.physics.diffsim``.

Python port of the internal C++ driver
(``mochi_physics/private/diffsim/test/mochi_differentiable_test.cpp``),
rebuilt so it runs from the open-source export without internal assets.

The diffsim contract implemented here (from the API docstrings):

- forward, per step: apply inputs -> capture pre-state -> ``Scene.step`` ->
  capture post-state (no input changes between the two captures);
- backward: ``reset_back_propagation`` once; accumulate the loss gradient with
  output-backward functions after ``prepare_back_propagate`` on the final state
  pair; then for each step, newest first: ``prepare_back_propagate`` ->
  ``back_propagate`` -> read per-step input gradients with ``set_*_backward``;
- gradients w.r.t. the initial pose/velocity are read once after the sweep.

Every analytic gradient is compared against a central finite difference of the
whole rollout loss, restarted from a captured initial state.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import superdex.physics as physics

diffsim = physics.diffsim

RIGID_POSE_SIZE = 7  # translation(3) + quaternion XYZW(4)
RIGID_DOF_SIZE = 6  # Lie tangent: d-translation(3) + d-rotation(3)
_D_TRANS = 3


@dataclasses.dataclass
class ActorEntry:
    """One dynamic actor and its offsets into the stacked vectors."""

    actor: object
    articulated: bool
    dofs_offset: int
    dofs_size: int
    pose_offset: int
    pose_size: int
    input_offset: int
    input_size: int
    force_dofs: list


def collect_actors(scene) -> list[ActorEntry]:
    """Enumerate dynamic actors like the C++ driver.

    Skips static and nested-link actors. Controller inputs exist only for
    articulated actors with a pose controller; external-force DoFs are the
    single-DoF joints of articulated actors (matching the C++ tests).
    """
    actors = []
    scene.for_each_actor(actors.append)
    entries: list[ActorEntry] = []
    total_dofs = total_pose = total_input = 0
    for actor in actors:
        if actor.is_static() or actor.is_nested_link_actor():
            continue
        articulated = actor.get_type() == physics.ActorType.ARTICULATED
        dofs = actor.get_num_dofs()
        pose = dofs if articulated else RIGID_POSE_SIZE
        has_controller = articulated and actor.has_articulated_pose_controller()
        inp = dofs if has_controller else 0
        force_dofs = []
        if articulated:
            info = actor.get_articulated_shape_info()
            for entry in info.dof_info:
                if entry.get_size() == 1:
                    force_dofs.append(entry.offset)
        entries.append(
            ActorEntry(
                actor,
                articulated,
                total_dofs,
                dofs,
                total_pose,
                pose,
                total_input,
                inp,
                force_dofs,
            )
        )
        total_dofs += dofs
        total_pose += pose
        total_input += inp
    return entries


def configure_for_differentiability(scene) -> None:
    """Make the scene differentiable and apply the C++ tests' solver settings."""
    diffsim.make_scene_differentiable(scene)

    bp_defaults = diffsim.BackPropagationSolverParams()
    sp = scene.get_solver_params()
    nl = sp.non_linear_solver
    nl.max_iter = 15
    nl.abs_tol = bp_defaults.outer_solver_abs_tol
    nl.rel_tol = bp_defaults.outer_solver_rel_tol
    nl.convergence_mode = bp_defaults.outer_solver_convergence_mode
    sp.non_linear_solver = nl
    ls = sp.linear_solver
    ls.abs_tol = bp_defaults.inner_solver_abs_tol
    sp.linear_solver = ls
    ee = sp.experimental_eval
    ee.fitted_saturation_hessian = physics.SaturationHessianParams.all(False)
    sp.experimental_eval = ee
    scene.set_solver_params(sp)

    dp = diffsim.get_back_propagation_solver_params(scene)
    dp.validate_finite_diff = True
    # The production default solves the adjoint only to abs_tol = 1e-3, which
    # leaves solve error of that order in small gradient components (measured:
    # 1.6e-2 relative error on a pendulum controller gradient, dropping to
    # 5.6e-5 with a tight tolerance). These tests measure gradient
    # correctness, so solve the adjoint tightly.
    dp.outer_solver_abs_tol = 1e-10
    dp.outer_solver_max_iter = 100
    diffsim.set_back_propagation_solver_params(scene, dp)


# ---------------------------------------------------------------------------
# State perturbation helpers (FD side), mirroring mochi_differentiable_test_utils.h
# ---------------------------------------------------------------------------


def rigid_raw_pose(actor) -> np.ndarray:
    tfm = actor.get_center_of_mass_transform()
    translation = np.asarray(tfm.translation, dtype=np.float64)
    quat = np.asarray(tfm.rotation.tolist(), dtype=np.float64)
    return np.concatenate([translation, quat])


def add_pose_eps(entry: ActorEntry, dof: int, eps: float) -> None:
    if entry.articulated:
        pose = np.zeros(entry.dofs_size)
        entry.actor.get_articulated_pose(pose)
        pose[dof] += eps
        entry.actor.set_articulated_pose_from_joints(pose)
        return
    raw = rigid_raw_pose(entry.actor)
    raw[dof] += eps
    if dof >= _D_TRANS:
        raw[3:] /= np.linalg.norm(raw[3:])
    entry.actor.set_center_of_mass_transform(
        physics.TransformRT(
            translation=raw[:3], rotation=physics.Quaternion(*raw[3:])
        )
    )


def add_vel_eps(entry: ActorEntry, dof: int, eps: float) -> None:
    if entry.articulated:
        vel = np.zeros(entry.dofs_size)
        entry.actor.get_articulated_joint_velocities(vel)
        vel[dof] += eps
        entry.actor.set_articulated_joint_velocities(vel)
        return
    lin = np.asarray(entry.actor.get_linear_velocity(), dtype=np.float64)
    ang = np.asarray(entry.actor.get_angular_velocity(), dtype=np.float64)
    delta = np.zeros(3)
    delta[dof % 3] = eps
    if dof < _D_TRANS:
        lin += delta
    else:
        ang += delta
    entry.actor.set_velocity(lin, ang)


# ---------------------------------------------------------------------------
# Losses (terminal, evaluated on the live scene state after the rollout)
# ---------------------------------------------------------------------------

POS_REF = np.array([1.4, -0.7, 0.5])
ROT_REF_VECTOR = np.array([-0.8, -0.2, 0.3])


class TranslationErrorLoss:
    """0.5 * || com_translation - ref ||^2 on one rigid actor (or link)."""

    def __init__(self, actor, ref=POS_REF):
        self.actor = actor
        self.ref = np.asarray(ref, dtype=np.float64)

    def value(self) -> float:
        pos = np.asarray(
            self.actor.get_center_of_mass_transform().translation, dtype=np.float64
        )
        disp = pos - self.ref
        return 0.5 * float(disp @ disp)

    def accumulate_output_grad(self) -> None:
        pos = np.asarray(
            self.actor.get_center_of_mass_transform().translation, dtype=np.float64
        )
        grad = np.zeros(RIGID_POSE_SIZE)
        grad[:3] = pos - self.ref
        diffsim.get_center_of_mass_transform_backward(self.actor, grad)


class QuaternionErrorLoss:
    """0.5 * || q - q_ref ||^2 on one rigid actor's orientation.

    The output-backward converts the quaternion-space gradient to the Lie
    tangent internally, which projects out the radial (norm) component; the FD
    side renormalizes after perturbing, which applies the same projection, so
    the two sides are consistent.
    """

    def __init__(self, actor, ref_rotation_vector=ROT_REF_VECTOR):
        self.actor = actor
        q_ref = physics.Quaternion.from_rotation_vector(
            np.asarray(ref_rotation_vector, dtype=np.float64)
        )
        self.q_ref = np.asarray(q_ref.tolist(), dtype=np.float64)

    def _quat(self) -> np.ndarray:
        return np.asarray(
            self.actor.get_center_of_mass_transform().rotation.tolist(),
            dtype=np.float64,
        )

    def value(self) -> float:
        diff = self._quat() - self.q_ref
        return 0.5 * float(diff @ diff)

    def accumulate_output_grad(self) -> None:
        grad = np.zeros(RIGID_POSE_SIZE)
        grad[3:] = self._quat() - self.q_ref
        diffsim.get_center_of_mass_transform_backward(self.actor, grad)


class ContactForceLoss:
    """0.5 * || total world contact force ||^2 on one rigid actor.

    Registers the ``TOTAL_CONTACT_FORCE`` query at construction time (it must
    be registered before the first step). The output-backward accumulates into
    per-contact force adjoints, so it exercises the contact-VJP path.
    """

    def __init__(self, actor):
        self.actor = actor
        actor.register_query(physics.QueryType.TOTAL_CONTACT_FORCE)

    def _force(self) -> np.ndarray:
        return np.asarray(self.actor.get_contact_force_world(), dtype=np.float64)

    def value(self) -> float:
        force = self._force()
        return 0.5 * float(force @ force)

    def accumulate_output_grad(self) -> None:
        diffsim.get_contact_force_world_backward(self.actor, self._force())


class ArticulatedPoseErrorLoss:
    """0.5 * || pose - ref ||^2 on one articulated actor's joint pose."""

    def __init__(self, actor, ref):
        self.actor = actor
        self.ref = np.asarray(ref, dtype=np.float64)

    def _pose(self) -> np.ndarray:
        pose = np.zeros(self.actor.get_num_dofs())
        self.actor.get_articulated_pose(pose)
        return pose

    def value(self) -> float:
        diff = self._pose() - self.ref
        return 0.5 * float(diff @ diff)

    def accumulate_output_grad(self) -> None:
        diffsim.get_articulated_pose_backward(self.actor, self._pose() - self.ref)


# ---------------------------------------------------------------------------
# The gradient-consistency case
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class GradientReport:
    name: str
    analytic: np.ndarray
    finite_diff: np.ndarray

    @property
    def rel_error(self) -> float:
        norm_a = np.linalg.norm(self.analytic)
        norm_b = np.linalg.norm(self.finite_diff)
        denom = max(norm_a, norm_b)
        if denom == 0.0:
            return 0.0
        return float(np.linalg.norm(self.analytic - self.finite_diff) / denom)

    def __str__(self) -> str:
        return (
            f"{self.name}: rel_error={self.rel_error:.3e}\n"
            f"  analytic = {np.array2string(self.analytic, precision=6)}\n"
            f"  fin.diff = {np.array2string(self.finite_diff, precision=6)}"
        )


class GradientCheckCase:
    """Run one scene through forward + adjoint + finite-difference validation."""

    def __init__(
        self,
        scene,
        losses,
        num_steps: int = 8,
        dt: float = 0.01,
        control_speed: float = 0.0,
        force_speed: float = 1.0,
        fd_eps: float = 1e-6,
    ):
        self.scene = scene
        self.losses = list(losses)
        self.num_steps = num_steps
        self.dt = dt
        self.fd_eps = fd_eps

        configure_for_differentiability(scene)
        self.entries = collect_actors(scene)
        self.total_dofs = sum(e.dofs_size for e in self.entries)
        self.total_pose = sum(e.pose_size for e in self.entries)
        self.total_input = sum(e.input_size for e in self.entries)

        # Control trajectory: initial pose of each controlled actor + a ramp.
        self.control = np.zeros((self.total_input, num_steps))
        for entry in self.entries:
            if entry.input_size > 0:
                pose = np.zeros(entry.input_size)
                entry.actor.get_articulated_pose(pose)
                self.control[
                    entry.input_offset : entry.input_offset + entry.input_size, 0
                ] = pose
        for j in range(num_steps):
            self.control[:, j] = self.control[:, 0] + control_speed * dt * j

        # External-force trajectory: a ramp on the valid force DoFs.
        self.forces = np.zeros((self.total_dofs, num_steps))
        for j in range(num_steps):
            amplitude = force_speed * dt * j
            for entry in self.entries:
                for dof in entry.force_dofs:
                    self.forces[entry.dofs_offset + dof, j] = amplitude

        self.state_init = scene.capture_state()

    # -- forward ------------------------------------------------------------

    def _rollout(self, capture: bool):
        pre, post = [], []
        for entry in self.entries:
            if entry.input_size > 0:
                entry.actor.set_articulated_target_velocity(
                    np.zeros(entry.input_size)
                )
        for i in range(self.num_steps):
            for entry in self.entries:
                if entry.input_size > 0:
                    target = self.control[
                        entry.input_offset : entry.input_offset + entry.input_size, i
                    ]
                    entry.actor.set_articulated_target_pose(
                        np.ascontiguousarray(target)
                    )
                if entry.force_dofs:
                    values = np.array(
                        [
                            self.forces[entry.dofs_offset + dof, i]
                            for dof in entry.force_dofs
                        ]
                    )
                    entry.actor.set_external_forces_on_dofs(
                        np.asarray(entry.force_dofs, dtype=np.int32), values
                    )
            if capture:
                pre.append(self.scene.capture_state())
            self.scene.step(self.dt)
            if capture:
                post.append(self.scene.capture_state())
        return pre, post

    def _loss_value(self) -> float:
        return sum(loss.value() for loss in self.losses)

    # -- adjoint ------------------------------------------------------------

    def run_backward(self):
        pre, post = self._rollout(capture=True)

        diffsim.reset_back_propagation(self.scene)
        diffsim.prepare_back_propagate(self.scene, post[-1], pre[-1])
        for loss in self.losses:
            loss.accumulate_output_grad()

        grad_control = np.zeros((self.total_input, self.num_steps))
        grad_force = np.zeros((self.total_dofs, self.num_steps))
        fd_valid_all = True
        max_residual = 0.0
        for i in range(self.num_steps, 0, -1):
            if i != self.num_steps:
                diffsim.prepare_back_propagate(self.scene, post[i - 1], pre[i - 1])
            diffsim.back_propagate(self.scene)
            # Accumulate diagnostics across the whole reverse sweep; the scene
            # stats only describe the most recent back_propagate call.
            step_stats = diffsim.get_back_propagation_scene_stats(self.scene)
            fd_valid_all = fd_valid_all and step_stats.finite_diff_valid
            max_residual = max(max_residual, step_stats.residual_norm)
            for entry in self.entries:
                if entry.input_size > 0:
                    g = np.zeros(entry.input_size)
                    diffsim.set_articulated_target_pose_backward(entry.actor, g)
                    grad_control[
                        entry.input_offset : entry.input_offset + entry.input_size,
                        i - 1,
                    ] = g
                if entry.force_dofs:
                    g = np.zeros(len(entry.force_dofs))
                    diffsim.set_external_forces_on_dofs_backward(
                        entry.actor, np.asarray(entry.force_dofs, dtype=np.int32), g
                    )
                    for k, dof in enumerate(entry.force_dofs):
                        grad_force[entry.dofs_offset + dof, i - 1] = g[k]

        grad_init_pose = np.zeros(self.total_pose)
        grad_init_vel = np.zeros(self.total_dofs)
        for entry in self.entries:
            if entry.articulated:
                gp = np.zeros(entry.dofs_size)
                diffsim.set_articulated_pose_from_joints_backward(entry.actor, gp)
                gv = np.zeros(entry.dofs_size)
                diffsim.set_articulated_joint_velocities_backward(entry.actor, gv)
                grad_init_pose[
                    entry.pose_offset : entry.pose_offset + entry.pose_size
                ] = gp
                grad_init_vel[
                    entry.dofs_offset : entry.dofs_offset + entry.dofs_size
                ] = gv
            else:
                gs = np.zeros(RIGID_POSE_SIZE)
                diffsim.set_center_of_mass_transform_backward(entry.actor, gs)
                gl = np.zeros(3)
                ga = np.zeros(3)
                diffsim.set_velocity_backward(entry.actor, gl, ga)
                grad_init_pose[
                    entry.pose_offset : entry.pose_offset + entry.pose_size
                ] = gs
                grad_init_vel[entry.dofs_offset : entry.dofs_offset + 3] = gl
                grad_init_vel[entry.dofs_offset + 3 : entry.dofs_offset + 6] = ga

        return {
            "control": grad_control,
            "force": grad_force,
            "init_pose": grad_init_pose,
            "init_vel": grad_init_vel,
            "fd_valid_all": fd_valid_all,
            "max_residual": max_residual,
        }

    # -- finite differences -------------------------------------------------

    def _fd_pair(self, perturb) -> float:
        """Central difference of the rollout loss around the initial state."""
        self.scene.restore_state(self.state_init, False)
        perturb(+self.fd_eps)
        self._rollout(capture=False)
        loss_p = self._loss_value()
        self.scene.restore_state(self.state_init, False)
        perturb(-self.fd_eps)
        self._rollout(capture=False)
        loss_m = self._loss_value()
        return (loss_p - loss_m) / (2.0 * self.fd_eps)

    def fd_init_pose(self) -> np.ndarray:
        out = np.zeros(self.total_pose)
        for entry in self.entries:
            for j in range(entry.pose_size):
                out[entry.pose_offset + j] = self._fd_pair(
                    lambda eps, e=entry, jj=j: add_pose_eps(e, jj, eps)
                )
        return out

    def fd_init_vel(self) -> np.ndarray:
        out = np.zeros(self.total_dofs)
        for entry in self.entries:
            for j in range(entry.dofs_size):
                out[entry.dofs_offset + j] = self._fd_pair(
                    lambda eps, e=entry, jj=j: add_vel_eps(e, jj, eps)
                )
        return out

    def fd_control_step(self, step: int) -> np.ndarray:
        out = np.zeros(self.total_input)
        ref = self.control[:, step].copy()

        for entry in self.entries:
            for j in range(entry.input_size):
                idx = entry.input_offset + j

                def perturb(eps, idx=idx, ref=ref, step=step):
                    self.control[:, step] = ref
                    self.control[idx, step] += eps

                out[idx] = self._fd_pair(perturb)
                self.control[:, step] = ref
        return out

    def fd_force_step(self, step: int) -> np.ndarray:
        indices = [
            entry.dofs_offset + dof
            for entry in self.entries
            for dof in entry.force_dofs
        ]
        out = np.zeros(len(indices))
        ref = self.forces[:, step].copy()
        for k, idx in enumerate(indices):

            def perturb(eps, idx=idx, ref=ref, step=step):
                self.forces[:, step] = ref
                self.forces[idx, step] += eps

            out[k] = self._fd_pair(perturb)
            self.forces[:, step] = ref
        return out

    def force_grad_at_step(self, grad_force: np.ndarray, step: int) -> np.ndarray:
        indices = [
            entry.dofs_offset + dof
            for entry in self.entries
            for dof in entry.force_dofs
        ]
        return grad_force[indices, step]

    # -- top level -----------------------------------------------------------

    def run(self, check_steps=None) -> list[GradientReport]:
        """Adjoint + FD sweep; returns one report per compared gradient block."""
        grads = self.run_backward()
        reports = [
            GradientReport("init_pose", grads["init_pose"], self.fd_init_pose()),
            GradientReport("init_vel", grads["init_vel"], self.fd_init_vel()),
        ]
        steps = range(self.num_steps) if check_steps is None else check_steps
        for i in steps:
            if self.total_input > 0:
                reports.append(
                    GradientReport(
                        f"control[step {i}]",
                        grads["control"][:, i],
                        self.fd_control_step(i),
                    )
                )
            if any(e.force_dofs for e in self.entries):
                reports.append(
                    GradientReport(
                        f"force[step {i}]",
                        self.force_grad_at_step(grads["force"], i),
                        self.fd_force_step(i),
                    )
                )
        self.fd_valid_all = grads["fd_valid_all"]
        self.max_residual = grads["max_residual"]
        self.scene.release_all_states()
        return reports
