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

"""Regressions of ``superdex_physics/examples/example_diffsim_tactile.py`` (loaded from the
source tree; skipped elsewhere): the nearest-face query is exact - a point inside a large
triangle is on the surface, and random meshes agree with an exhaustive point-to-triangle
reference - and the taxel-map loss handles pads without contact: no pad in contact gives a
graph-free loss whose vector-Jacobian product is zero (the unconditional ``backward`` used
to raise), one pad in contact seeds its force cotangent, which a central finite difference of
the map confirms, and contact removed then regained seeds nothing stale."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import unittest

import numpy as np

EXAMPLE = pathlib.Path(__file__).resolve().parents[3] / "examples" / "example_diffsim_tactile.py"

try:
    import superdex.physics as physics  # noqa: F401  (the example imports the native modules)
    import torch
    import trimesh
except ImportError as error:  # pragma: no cover - the native build or a dependency is absent
    if os.environ.get("SUPERDEX_REQUIRE_NATIVE"):
        raise
    raise unittest.SkipTest(f"the tactile example tests need the native build, torch and trimesh: {error}") from error

if not EXAMPLE.is_file():  # pragma: no cover - installed wheel without the source tree
    raise unittest.SkipTest(f"the tactile example is not beside this checkout: {EXAMPLE}")


def _load_example():
    name = "example_diffsim_tactile"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, EXAMPLE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


class NearestFacesTest(unittest.TestCase):
    def test_point_inside_a_large_triangle_is_on_the_surface(self) -> None:
        """One triangle of 1 m: a point deep inside it lies 0.4 m from the centroid, so a
        candidate filter by centroid distance (4 mm until 2026-09-10) found no face."""
        example = _load_example()
        mesh = trimesh.Trimesh([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0, 1, 2]], process=False)
        points = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.001], [2.0, 0.0, 0.0]])
        dist, face = example.nearest_faces(mesh, points)
        np.testing.assert_allclose(dist, [0.0, 0.001, 1.0], atol=1e-15)
        self.assertEqual(face.tolist(), [0, 0, 0])
        with self.assertRaisesRegex(ValueError, "no face"):
            example.nearest_faces(trimesh.Trimesh(process=False), points)

    def test_matches_an_exhaustive_reference_on_random_meshes(self) -> None:
        example = _load_example()
        rng = np.random.default_rng(7)
        for trial in range(4):
            with self.subTest(trial=trial):
                mesh = trimesh.convex.convex_hull(rng.normal(size=(30, 3)) * rng.uniform(0.2, 2.0, 3))
                points = rng.uniform(-3.0, 3.0, (40, 3))
                points[:10] = mesh.triangles[:10].mean(axis=1)  # on the surface
                dist, face = example.nearest_faces(mesh, points)
                _, reference, _ = trimesh.proximity.closest_point_naive(mesh, points)
                np.testing.assert_allclose(dist, reference, rtol=0.0, atol=1e-12)
                np.testing.assert_allclose(dist[:10], 0.0, atol=1e-12)
                on_face = trimesh.triangles.closest_point(mesh.triangles[face], points)
                np.testing.assert_allclose(np.linalg.norm(on_face - points, axis=1), dist, atol=1e-12)


class _Transform:
    """The identity root transform of a distal link (link frame = world frame)."""

    rotation = np.array([0.0, 0.0, 0.0, 1.0])
    translation = np.zeros(3)


class _Handle:
    def __init__(self, value: int):
        self.value = value

    def __eq__(self, other) -> bool:
        return isinstance(other, _Handle) and other.value == self.value

    def __hash__(self) -> int:
        return hash(self.value)


class _Actor:
    def __init__(self, name: str, handle: int):
        self._name = name
        self._handle = _Handle(handle)
        self.rows: list = []

    def get_name(self) -> str:
        return self._name

    def get_handle(self) -> _Handle:
        return self._handle

    def get_root_transform(self) -> _Transform:
        return _Transform()

    def get_contact_points_world(self) -> list:
        return self.rows


class _Row:
    """A contact of the pad's own sample against the object (the supported direction)."""

    def __init__(self, pad: _Actor, obj: _Actor, position, outward, force):
        self.actor_a = pad.get_handle()
        self.actor_b = obj.get_handle()
        self.pos_a = np.asarray(position, dtype=np.float64)
        self.normal = -np.asarray(outward, dtype=np.float64)  # points away from the object
        self.pos_b = self.pos_a
        self.force = np.asarray(force, dtype=np.float64)


class _Rig:
    """The parts of ``Rig`` the taxel readout uses: one distal link per finger, all at the
    identity transform, and the real taxel layout."""

    def __init__(self, example):
        self.layout = example.Layout("right")
        self.distal = {f: _Actor(f"right_hand_{example.LINK2[f]}", 10 + i) for i, f in enumerate(example.FINGERS)}
        self.parts = {f: [self.distal[f]] for f in example.FINGERS}
        self.hand_handles = {a.get_handle() for a in self.distal.values()}


class TaxelMapLossTest(unittest.TestCase):
    """The adjoint seeding is observed through the example's ``diffsim`` calls."""

    def setUp(self) -> None:
        self.example = _load_example()
        self.rig = _Rig(self.example)
        self.object = _Actor("object", 99)
        self.seeded: list[tuple[str, str, np.ndarray]] = []
        example = self.example
        real = (example.diffsim.get_contact_points_backward, example.diffsim.get_root_transform_backward)

        def contact_backward(actor, grad):
            self.seeded.append(("contact", actor.get_name(), np.array(grad, dtype=np.float64)))

        def root_backward(actor, grad):
            self.seeded.append(("root", actor.get_name(), np.array(grad, dtype=np.float64)))

        example.diffsim.get_contact_points_backward = contact_backward
        example.diffsim.get_root_transform_backward = root_backward
        self.addCleanup(setattr, example.diffsim, "get_contact_points_backward", real[0])
        self.addCleanup(setattr, example.diffsim, "get_root_transform_backward", real[1])

    def _touch(self, finger: str, taxel: int, magnitude: float) -> None:
        """A contact sample at a taxel of the finger's pad, pressed along the taxel's inward normal."""
        fi = self.example.FINGERS.index(finger)
        position = self.rig.layout.pos[fi][taxel]
        outward = self.rig.layout.normal[fi][taxel]
        self.rig.distal[finger].rows = [_Row(self.rig.distal[finger], self.object, position, outward, -magnitude * outward)]

    def test_no_contact_is_a_zero_vector_jacobian_product(self) -> None:
        example = self.example
        field = example.TaxelField(self.rig)
        target = np.zeros((5, 120))
        target[1, 5] = 1.0
        loss = example.TaxelMapLoss(field, target)
        np.testing.assert_array_equal(field.numpy(), 0.0)
        self.assertEqual(loss.value(), 0.5)
        loss.accumulate_output_grad()  # a graph-free loss: nothing to seed, no exception
        self.assertEqual(self.seeded, [])

    def test_one_pad_in_contact_seeds_its_force_cotangent(self) -> None:
        example = self.example
        field = example.TaxelField(self.rig)
        loss = example.TaxelMapLoss(field, np.zeros((5, 120)))
        self._touch("index", 40, 2.0)
        n = field.numpy()
        self.assertGreater(n[1].max(), 0.0)
        self.assertEqual(np.count_nonzero(n[[0, 2, 3, 4]]), 0, "only the touched pad reads a force")
        loss.accumulate_output_grad()
        contact = [entry for entry in self.seeded if entry[0] == "contact"]
        self.assertEqual([entry[1] for entry in contact], ["right_hand_index_rota_link2"])
        cotangent = contact[0][2]
        self.assertEqual(cotangent.shape, (3,))
        self.assertGreater(np.linalg.norm(cotangent), 0.0)
        # Central finite differences of the loss in the reported force (the map is smooth
        # where the touched taxels are compressed).
        row = self.rig.distal["index"].rows[0]
        eps = 1e-6
        fd = np.zeros(3)
        for k in range(3):
            values = []
            for sign in (1.0, -1.0):
                row.force[k] += sign * eps
                values.append(loss.value())
                row.force[k] -= sign * eps
            fd[k] = (values[0] - values[1]) / (2.0 * eps)
        np.testing.assert_allclose(cotangent, fd, rtol=1e-6, atol=1e-9)
        roots = [entry for entry in self.seeded if entry[0] == "root"]
        self.assertEqual([entry[1] for entry in roots], ["right_hand_index_rota_link2"])
        self.assertEqual(roots[0][2].shape, (7,))
        np.testing.assert_array_equal(roots[0][2][:3], 0.0, "the readout does not depend on the link position")

    def test_contact_removed_then_regained_seeds_nothing_stale(self) -> None:
        example = self.example
        field = example.TaxelField(self.rig)
        loss = example.TaxelMapLoss(field, np.zeros((5, 120)))
        self._touch("thumb", 60, 1.0)
        loss.accumulate_output_grad()
        first = [entry for entry in self.seeded if entry[0] == "contact"][0][2].copy()
        self.seeded.clear()
        self.rig.distal["thumb"].rows = []
        self.assertEqual(loss.value(), 0.0)
        loss.accumulate_output_grad()
        self.assertEqual(self.seeded, [], "no contact: nothing to seed")
        self._touch("thumb", 60, 3.0)
        loss.accumulate_output_grad()
        contact = [entry for entry in self.seeded if entry[0] == "contact"]
        self.assertEqual(len(contact), 1)
        np.testing.assert_allclose(contact[0][2], 3.0 * first, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
