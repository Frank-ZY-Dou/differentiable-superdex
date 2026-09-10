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

"""Regressions of ``SoftMonitor`` in ``superdex_physics/examples/example_diffsim_video.py``
(loaded from the source tree; skipped elsewhere): every geometry read is validated before it
is reduced - a non-finite displacement, position or determinant is refused whichever body
carries it, a Python ``min`` never hides it - a rejected read blocks the rollout's verdict
until the next ``begin()``, an inverted element is refused in the initial state and by the
policy after a step, invalid construction inputs are rejected, and a replay observation that
fails leaves the two substep captures to ``step_with_substeps``, which releases each once.

The bodies are stubs whose displacements the tests set; the contact checker inside the
monitor watches a real scene (a soft cube on a plane), stepped once so its query has data."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import unittest
from unittest import mock

import numpy as np

EXAMPLE = pathlib.Path(__file__).resolve().parents[3] / "examples" / "example_diffsim_video.py"

try:
    import superdex.physics as physics
except ImportError as error:  # pragma: no cover - the native build is absent
    if os.environ.get("SUPERDEX_REQUIRE_NATIVE"):
        raise
    raise unittest.SkipTest(f"the video example tests need the native build: {error}") from error

if not EXAMPLE.is_file():  # pragma: no cover - installed wheel without the source tree
    raise unittest.SkipTest(f"the video example is not beside this checkout: {EXAMPLE}")


def _load_example():
    """The example as a module; a failed load leaves nothing behind in ``sys.modules``."""
    name = "example_diffsim_video"
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


try:
    video = _load_example()
except ImportError as error:  # pragma: no cover - a dependency of the example is absent
    raise unittest.SkipTest(f"the video example cannot be imported here: {error}") from error

from .diffsim import scenes  # noqa: E402
from .diffsim.harness import configure_for_differentiability  # noqa: E402

DT = 0.01


def setUpModule() -> None:
    physics.initialize(num_worker_threads=0)


def tearDownModule() -> None:
    physics.shutdown()


class _Body:
    """A soft body whose displacements the test controls."""

    def __init__(self, name: str, num_nodes: int):
        self._name = name
        self.displacements = np.zeros(3 * num_nodes)

    def get_name(self) -> str:
        return self._name

    def get_displacements(self) -> np.ndarray:
        return self.displacements


class SoftMonitorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scene, self.jelly = scenes.soft_cube_on_plane("rich")
        self.addCleanup(physics.destroy_scene, self.scene)
        # As the demos: a differentiable scene with tight solvers, so a plain step converges
        # below SOFT_RESIDUAL_TOLERANCE and a replay step is not split.
        configure_for_differentiability(self.scene)
        video.tighten_solvers(self.scene)
        coordinates, connectivity = video.box_tet_mesh(size=0.1, cells=2)  # flat, as the engine takes them
        self.coordinates = coordinates
        self.connectivity = connectivity.reshape(-1, 4)
        self.num_nodes = self.coordinates.size // 3

    def _body(self, name: str = "body") -> _Body:
        return _Body(name, self.num_nodes)

    def _monitor(self, *bodies: _Body, limit: float = 0.02):
        return video.SoftMonitor(
            self.scene, {body: (self.coordinates, self.connectivity) for body in bodies}, limit
        )

    def _observed_rollout(self, monitor, num_steps: int = 2) -> None:
        monitor.begin()
        monitor.observe_initial()
        for step in range(num_steps):
            self.scene.step(DT)
            monitor.observe(step, DT)

    def test_construction_rejects_invalid_inputs(self) -> None:
        body = self._body()
        with self.assertRaisesRegex(ValueError, "no soft body"):
            video.SoftMonitor(self.scene, {}, 0.02)
        for limit in (float("nan"), float("inf"), -0.01):
            with self.assertRaisesRegex(ValueError, "penetration limit"):
                self._monitor(body, limit=limit)
        bad = {
            "rest coordinates must": (np.zeros((0, 3)), self.connectivity),
            "non-finite rest": (np.where(np.arange(self.coordinates.size) == 4, np.nan, self.coordinates), self.connectivity),
            "connectivity must": (self.coordinates, np.zeros((0, 4), int)),
            "integer": (self.coordinates, self.connectivity.astype(float)),
            "index out of range": (self.coordinates, np.where(self.connectivity == 0, self.num_nodes, self.connectivity)),
            "non-positive or non-finite rest volume": (self.coordinates, self.connectivity[:, [0, 2, 1, 3]]),
        }
        for message, (coordinates, connectivity) in bad.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    video.SoftMonitor(self.scene, {body: (coordinates, connectivity)}, 0.02)
        body.displacements[7] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite node positions"):
            self._monitor(body)
        body.displacements[:] = 0.0
        body.displacements[2::3] = -2.0 * self.coordinates.reshape(-1, 3)[:, 2]  # mirrored: every element inverted
        with self.assertRaisesRegex(RuntimeError, "inverted or degenerate element in the state the monitor is built on"):
            self._monitor(body)
        body.displacements = np.zeros(3 * self.num_nodes - 3)
        with self.assertRaisesRegex(ValueError, "displacement values for"):
            self._monitor(body)

    def test_non_finite_geometry_is_refused_and_latched(self) -> None:
        """A NaN displacement after a good observation raises, the rollout gets no verdict
        even after good states follow, and only ``begin()`` clears the latch; an overflow to
        an infinite determinant is refused the same way."""
        body = self._body()
        monitor = self._monitor(body)
        self._observed_rollout(monitor)
        monitor.check("clean")
        self.assertEqual(monitor.rollouts, 1)
        monitor.begin()
        monitor.observe_initial()
        self.scene.step(DT)
        monitor.observe(0, DT)
        body.displacements[5] = float("nan")
        self.scene.step(DT)
        with self.assertRaisesRegex(ValueError, "body: non-finite node positions"):
            monitor.observe(1, DT)
        body.displacements[5] = 0.0
        self.scene.step(DT)
        monitor.observe(2, DT)
        with self.assertRaisesRegex(RuntimeError, "1 geometry reads were rejected"):
            monitor.check("latched")
        self.assertEqual(monitor.rollouts, 1)
        self.assertIn("1 geometry reads rejected", monitor.summary())
        self._observed_rollout(monitor)
        monitor.check("after begin")
        self.assertEqual(monitor.rollouts, 2)
        body.displacements[:] = 1e120 * self.coordinates  # the edge products overflow to infinity
        with self.assertRaisesRegex(ValueError, "non-finite element determinant"):
            monitor.observe_initial()
        self.assertEqual(monitor.rejected_total, 2)
        body.displacements[:] = 1e200  # every node rounds to the same point: a collapsed element
        with self.assertRaisesRegex(RuntimeError, "degenerate element in the initial state"):
            monitor.observe_initial()
        self.assertEqual(monitor.rejected_total, 3)

    def test_one_bad_body_is_not_hidden_by_a_good_one(self) -> None:
        for bad_first in (True, False):
            with self.subTest(bad_first=bad_first):
                good, bad = self._body("good"), self._body("bad")
                monitor = self._monitor(*((bad, good) if bad_first else (good, bad)))
                self._observed_rollout(monitor, num_steps=1)
                bad.displacements[0] = float("inf")
                self.scene.step(DT)
                with self.assertRaisesRegex(ValueError, "bad: non-finite node positions"):
                    monitor.observe(1, DT)
                with self.assertRaisesRegex(RuntimeError, "rejected"):
                    monitor.check("masked")

    def test_inverted_element_is_refused(self) -> None:
        """A finite but inverted element: refused in the initial state, and by the policy
        when it appears after a step (the observation itself succeeds)."""
        body = self._body()
        monitor = self._monitor(body)
        mirrored = -2.0 * self.coordinates.reshape(-1, 3)[:, 2]
        monitor.begin()
        body.displacements[2::3] = mirrored
        with self.assertRaisesRegex(RuntimeError, "inverted or degenerate element in the initial state"):
            monitor.observe_initial()
        with self.assertRaisesRegex(RuntimeError, "rejected"):
            monitor.check("inverted start")
        body.displacements[:] = 0.0
        monitor.begin()
        monitor.observe_initial()
        self.scene.step(DT)
        body.displacements[2::3] = mirrored
        monitor.observe(0, DT)
        self.assertLess(monitor.min_det, 0.0)
        with self.assertRaisesRegex(RuntimeError, "inverted or degenerate element \\(min deformation-gradient"):
            monitor.check("inverted after a step")

    def test_verdict_requires_the_initial_state_and_a_step(self) -> None:
        body = self._body()
        monitor = self._monitor(body)
        monitor.begin()
        with self.assertRaisesRegex(RuntimeError, "initial state was not observed"):
            monitor.check("no initial")
        monitor.observe_initial()
        with self.assertRaisesRegex(RuntimeError, "no \\(sub\\)step was observed"):
            monitor.check("no step")
        self.scene.step(DT)
        monitor.observe(0, DT)
        monitor.check("complete")
        self.assertEqual((monitor.rollouts, monitor.observed_total), (1, 1))
        self.assertGreater(monitor.worst_depth, 0.0, "the jelly rests on the plane")

    def test_replay_observation_failure_releases_each_capture_once(self) -> None:
        """``step()`` observes before it releases the substep captures: when the observation
        raises, ``step_with_substeps`` releases the two captures it still owns, each exactly
        once, and the observation's own error is what propagates."""
        if not physics.uses_double_precision():
            raise unittest.SkipTest("the demo's substep tolerance is a double-precision figure (float32 floors above it)")
        body = self._body()
        monitor = self._monitor(body)
        monitor.begin()
        monitor.observe_initial()
        captured: list = []
        released: list = []
        real_capture = type(self.scene).capture_state
        real_release = type(self.scene).release_state

        def capture(scene):
            handle = real_capture(scene)
            captured.append(handle.value)
            return handle

        def release(scene, handle):
            released.append(handle.value)
            return real_release(scene, handle)

        body.displacements[1] = float("nan")
        with mock.patch.object(type(self.scene), "capture_state", capture):
            with mock.patch.object(type(self.scene), "release_state", release):
                with self.assertRaisesRegex(ValueError, "body: non-finite node positions"):
                    monitor.step(DT, 0)
        self.assertEqual(len(captured), 2, "the first accepted substep: a pre and a post capture")
        self.assertEqual(sorted(released), sorted(captured), "each capture released once, by step_with_substeps")
        self.assertEqual(monitor.invalid, 1)
        body.displacements[1] = 0.0
        monitor.begin()
        monitor.observe_initial()
        released.clear()
        captured.clear()
        with mock.patch.object(type(self.scene), "capture_state", capture):
            with mock.patch.object(type(self.scene), "release_state", release):
                monitor.step(DT, 0)
        self.assertEqual(sorted(released), sorted(captured), "each capture released once, by the observer")
        self.assertGreaterEqual(len(captured), 2)
        monitor.check("replay")


if __name__ == "__main__":
    unittest.main()
