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

from __future__ import annotations

import unittest

import superdex.robotics as sdr
from superdex.robotics import BotPrefab


class SuperdexRoboticsTest(unittest.TestCase):
    def test_import_smoke(self) -> None:
        self.assertTrue(hasattr(sdr, "BotPrefab"))
        self.assertTrue(hasattr(sdr, "create_context"))
        self.assertIs(BotPrefab, sdr.BotPrefab)


class UrdfInertialImportTest(unittest.TestCase):
    """The importer expresses a link's inertia tensor in the link frame: URDF gives it in the
    inertial frame, rotated from the link by the rpy of <inertial><origin>."""

    URDF = """<?xml version="1.0"?>
<robot name="inertial_test">
  <link name="rotated">
    <inertial>
      <origin xyz="0.01 0.02 0.03" rpy="0.3 -0.4 0.5"/>
      <mass value="2.0"/>
      <inertia ixx="1.0" ixy="0.1" ixz="-0.2" iyy="2.0" iyz="0.05" izz="3.0"/>
    </inertial>
  </link>
  <link name="plain">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="1.0" ixy="0.1" ixz="-0.2" iyy="2.0" iyz="0.05" izz="3.0"/>
    </inertial>
  </link>
  <joint name="weld" type="fixed">
    <parent link="rotated"/>
    <child link="plain"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
  </joint>
</robot>
"""

    @staticmethod
    def _rotation(roll, pitch, yaw):
        import math

        cr, sr, cp, sp, cy, sy = (
            math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw))
        rx = [[1, 0, 0], [0, cr, -sr], [0, sr, cr]]
        ry = [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]
        rz = [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]
        mult = lambda a, b: [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        return mult(rz, mult(ry, rx))   # URDF fixed-axis rpy: R = Rz(yaw) Ry(pitch) Rx(roll)

    def test_rotated_inertial_lands_in_the_link_frame(self) -> None:
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "inertial_test.urdf")
            with open(path, "w") as handle:
                handle.write(self.URDF)
            prefab = sdr.load_bot_prefab_from_urdf_file(path)
        links = {prefab.links[i].name: prefab.links[i] for i in range(len(prefab.links))}
        tensor = [[1.0, 0.1, -0.2], [0.1, 2.0, 0.05], [-0.2, 0.05, 3.0]]
        rot = self._rotation(0.3, -0.4, 0.5)
        mult = lambda a, b: [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        transposed = [[rot[j][i] for j in range(3)] for i in range(3)]
        expected = mult(rot, mult(tensor, transposed))     # R I R^T
        import superdex.physics as physics

        tolerance = 1e-12 if physics.uses_double_precision() else 1e-6   # the single build stores floats
        got = list(links["rotated"].moment_of_inertia)
        for value, index in zip(got, ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))):
            self.assertAlmostEqual(value, expected[index[0]][index[1]], delta=tolerance)
        for value, reference in zip(links["rotated"].center_of_mass, (0.01, 0.02, 0.03)):
            self.assertAlmostEqual(value, reference, delta=tolerance)
        self.assertAlmostEqual(links["rotated"].mass, 2.0, delta=tolerance)
        for value, reference in zip(links["plain"].moment_of_inertia, (1.0, 0.1, -0.2, 2.0, 0.05, 3.0)):
            self.assertAlmostEqual(value, reference, delta=tolerance)
