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

"""A/B comparison: finite-difference vs analytic Hessian-vector products in the
adjoint outer solve (``BackPropagationSolverParams.use_analytic_hvp``).

For each scene, runs the full gradient check in both modes and reports the
worst gradient error against rollout finite differences, the direct
disagreement between the two modes' gradients, and the summed adjoint solve
time. Run from ``superdex_physics/wheels/superdex-physics`` with a source build that has the flag::

    SUPERDEX_PRECISION=double python -m test.diffsim.ab_compare
"""

from __future__ import annotations

import numpy as np
import superdex.physics as physics

from . import scenes
from .harness import (
    ArticulatedPoseErrorLoss,
    GradientCheckCase,
    TranslationErrorLoss,
    diffsim,
)

REPEATS = 20  # timing repeats of the backward sweep


def build_cases():
    def rigid_coulomb():
        scene, cube = scenes.rigid_on_plane("coulomb")
        return scene, [TranslationErrorLoss(cube)], {}

    def two_cubes():
        scene, bottom = scenes.two_cubes_on_plane("coulomb")
        return scene, [TranslationErrorLoss(bottom)], {}

    def controller():
        scene, chain = scenes.pendulum(with_controller=True)
        return (
            scene,
            [ArticulatedPoseErrorLoss(chain, np.array([0.4, -0.2]))],
            {"control_speed": 0.5},
        )

    def free_chain():
        scene, chain = scenes.free_chain_on_plane("coulomb")
        ref = np.zeros(chain.get_num_dofs())
        ref[-1] = 0.3
        return scene, [ArticulatedPoseErrorLoss(chain, ref)], {}

    def mesh_box():
        scene, cube = scenes.rigid_on_mesh_box("coulomb")
        return scene, [TranslationErrorLoss(cube)], {}

    # Rigid and articulated scenes only: GradientCheckCase's initial-state blocks use the
    # rigid and articulated accessors (a soft actor's nodal state is covered by
    # test_diffsim_soft with its own finite differences).
    return {
        "rigid_on_plane_coulomb": rigid_coulomb,
        "two_cubes_on_plane": two_cubes,
        "pendulum_controller": controller,
        "free_chain_on_plane": free_chain,
        "rigid_on_mesh_box_coulomb": mesh_box,
    }


def run_mode(build, analytic: bool):
    scene, losses, kwargs = build()
    case = GradientCheckCase(scene, losses, **kwargs)
    dp = diffsim.get_back_propagation_solver_params(scene)
    dp.use_analytic_hvp = analytic
    diffsim.set_back_propagation_solver_params(scene, dp)

    reports = case.run()
    worst = max(reports, key=lambda r: r.rel_error)

    # Timing: repeat the backward sweep (the trajectory simply continues from
    # wherever the previous rollout ended; solve_time isolates the adjoint
    # solver, so the drifting state only varies the problem slightly).
    solve_times = []
    for _ in range(REPEATS):
        grads = case.run_backward()
        solve_times.append(grads["solve_time"])
        scene.release_all_states()
    result = {
        "worst_rel_err": worst.rel_error,
        "worst_block": worst.name,
        "fd_valid": case.fd_valid_all,
        "max_residual": case.max_residual,
        "solve_time_ms": 1e3 * float(np.median(solve_times)),
        "grads": {k: grads[k] for k in ("control", "force", "init_pose", "init_vel")},
    }
    physics.destroy_scene(scene)
    return result


def cross_diff(a, b) -> float:
    worst = 0.0
    for key in a["grads"]:
        ga, gb = a["grads"][key], b["grads"][key]
        denom = max(np.linalg.norm(ga), np.linalg.norm(gb))
        if denom > 0:
            worst = max(worst, float(np.linalg.norm(ga - gb) / denom))
    return worst


def main() -> None:
    assert physics.uses_double_precision(), "run with SUPERDEX_PRECISION=double"
    physics.initialize(num_worker_threads=0)
    header = (
        f"{'scene':26s} {'mode':9s} {'worst_rel_err':>13s} "
        f"{'fd_valid':>8s} {'solve_ms':>9s}   worst block"
    )
    print(header)
    print("-" * len(header))
    for name, build in build_cases().items():
        results = {}
        for analytic in (False, True):
            mode = "analytic" if analytic else "fd"
            r = run_mode(build, analytic)
            results[mode] = r
            print(
                f"{name:26s} {mode:9s} {r['worst_rel_err']:13.3e} "
                f"{str(r['fd_valid']):>8s} {r['solve_time_ms']:9.3f}   {r['worst_block']}"
            )
        print(
            f"{'':26s} {'x-diff':9s} {cross_diff(results['fd'], results['analytic']):13.3e}"
            "   (fd vs analytic gradients)"
        )
    physics.shutdown()


if __name__ == "__main__":
    main()
