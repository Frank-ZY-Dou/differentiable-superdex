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

"""Single-precision gradients against a double-precision reference.

The finite-difference checks of this directory need double precision (at float32 the
difference quotients are as noisy as the gradients they check), so they leave the
single-precision build uncovered. This module covers it with a ground truth that does not
depend on finite differences: the gradients of a fixed set of rollouts computed by the
double-precision build, stored in ``data/fp64_reference_gradients.json``.

- On the double-precision build the test checks that the stored reference is current
  (relative agreement 1e-9): an engine change that moves a gradient fails here and asks
  for a regeneration, which is a deliberate act.
- On the single-precision build the test checks every gradient block against the
  reference at ``FP32_TOLERANCE``, measured on 2026-09-05 (see the constant).

Regenerate the reference (from this directory, on a double-precision build)::

    SUPERDEX_PRECISION=double python -m test.diffsim.test_precision_reference --regenerate
"""

from __future__ import annotations

import json
import pathlib
import sys
import unittest
from unittest import mock

import numpy as np
import superdex.physics as physics

from .ab_compare import build_cases
from .harness import GradientCheckCase

REFERENCE_PATH = pathlib.Path(__file__).with_name("data") / "fp64_reference_gradients.json"
BLOCKS = ("control", "force", "init_pose", "init_vel")
# Relative error of the single-precision gradients against the double-precision reference,
# per gradient block. Measured on 2026-09-06 (engine at 1.0.0+diffsim.1): pendulum with a
# controller 2e-5..1.2e-4, a cube sliding on the ground 1.2e-4..1.8e-4, a free chain on the
# ground 7e-5..2.5e-3, a cube sliding off-center on a static mesh-collider box 3.2e-3 (initial
# poses) and 4.1e-3 (initial velocities), two stacked cubes 2.9e-3 (initial velocities) and
# 9.8e-3 (initial poses) - the stacked contact is where float32 residuals and the 1e-5 Newton
# tolerance of that build bite hardest. The tolerance is three times the worst block: a wrong
# adjoint term moves a gradient by O(1) (every engine bug found so far did), a precision loss
# by these fractions of a percent.
FP32_TOLERANCE = 3e-2
FP64_TOLERANCE = 1e-9


def compute_gradients() -> dict[str, dict[str, list]]:
    """The gradients of every case, as nested lists (JSON-ready), keyed by case and block."""
    result = {}
    for name, build in build_cases().items():
        scene, losses, kwargs = build()
        try:
            case = GradientCheckCase(scene, losses, **kwargs)
            grads = case.run_backward()
            result[name] = {block: np.asarray(grads[block], dtype=np.float64).tolist() for block in BLOCKS}
        finally:
            scene.release_all_states()
            physics.destroy_scene(scene)
    return result


def _relative_error(actual, reference) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    denom = np.linalg.norm(reference)
    if denom == 0.0:
        return float(np.linalg.norm(actual))
    return float(np.linalg.norm(actual - reference) / denom)


class PrecisionReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        physics.initialize(num_worker_threads=0)
        if not REFERENCE_PATH.is_file():
            raise unittest.SkipTest(f"no reference at {REFERENCE_PATH}; regenerate it on a fp64 build")
        cls.reference = json.loads(REFERENCE_PATH.read_text())

    @classmethod
    def tearDownClass(cls) -> None:
        physics.shutdown()

    def test_gradients_match_the_double_precision_reference(self) -> None:
        tolerance = FP64_TOLERANCE if physics.uses_double_precision() else FP32_TOLERANCE
        actual = compute_gradients()
        self.assertEqual(set(actual), set(self.reference), "the case set changed: regenerate the reference")
        errors = {}
        for name, blocks in actual.items():
            for block in BLOCKS:
                errors[(name, block)] = _relative_error(blocks[block], self.reference[name][block])
        worst = max(errors, key=errors.get)
        self.assertLessEqual(
            errors[worst],
            tolerance,
            f"{worst[0]}/{worst[1]}: relative error {errors[worst]:.2e} > {tolerance:.0e}"
            + ("" if physics.uses_double_precision() else " (single precision)")
            + f"; all: {{{', '.join(f'{k[0]}/{k[1]}: {v:.1e}' for k, v in sorted(errors.items()))}}}",
        )


def main() -> None:
    if "--regenerate" not in sys.argv:
        print(__doc__)
        return
    if not physics.uses_double_precision():
        raise SystemExit("regenerate the reference on the double-precision build (SUPERDEX_PRECISION=double)")
    physics.initialize(num_worker_threads=0)
    try:
        reference = compute_gradients()
    finally:
        physics.shutdown()
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_PATH.write_text(json.dumps(reference, indent=1, sort_keys=True) + "\n")
    print(f"wrote {REFERENCE_PATH}: {len(reference)} cases")


if __name__ == "__main__":
    main()


class TorchBridgePrecisionContractTest(unittest.TestCase):
    """``diffsim_torch`` presents the engine's results as float64 tensors, so both bridges refuse
    the single-precision engine at construction (before 2026-09-07: a float64
    tensor drove the float32 engine and float32-accurate gradients came back as float64, with
    ``fd_valid`` set). On the single-precision build this is the real refusal; on the
    double-precision build the check is exercised through a patched precision query, and the
    real query admits."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(f"torch is not installed: {exc}")
        physics.initialize(num_worker_threads=0)

    @classmethod
    def tearDownClass(cls) -> None:
        physics.shutdown()

    def _constructors(self, scene, cube):
        from superdex.physics import diffsim_torch
        from .harness import TranslationErrorLoss

        import torch

        loss = TranslationErrorLoss(cube)
        return [
            lambda: diffsim_torch.TorchRollout(
                scene, dt=0.01, num_steps=2, force_actors=[cube], terminal_losses=[loss]
            ),
            lambda: diffsim_torch.PolicyRollout(
                scene,
                dt=0.01,
                num_steps=2,
                policy=torch.nn.Linear(3, 6).double(),
                observations=[diffsim_torch.TranslationObservation(cube)],
                force_actors=[cube],
                terminal_losses=[loss],
            ),
        ]

    def test_bridges_refuse_the_single_precision_engine(self) -> None:
        from . import scenes

        scene, cube = scenes.rigid_free()
        self.addCleanup(physics.destroy_scene, scene)
        physics.diffsim.make_scene_differentiable(scene)
        constructors = self._constructors(scene, cube)
        if physics.uses_double_precision():
            for construct in constructors:
                bridge = construct()
                bridge.close()
            with mock.patch.object(physics, "uses_double_precision", return_value=False):
                for construct in constructors:
                    with self.assertRaisesRegex(RuntimeError, "double-precision engine"):
                        construct()
        else:
            for construct in constructors:
                with self.assertRaisesRegex(RuntimeError, "double-precision engine"):
                    construct()
