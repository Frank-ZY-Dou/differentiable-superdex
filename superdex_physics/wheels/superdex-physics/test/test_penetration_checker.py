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

"""``superdex.physics.utils.penetration.PenetrationChecker`` against known contact states:
a cube resting on the ground under a penalty contact overlaps it by less than the penalty's
smoothing distance and by less the stiffer the contact, a free cube shows no contact, two
stacked cubes report the cube-cube and cube-ground pairs, an articulated link pushing a
cube names the link, and a state restore keeps the query alive."""

from __future__ import annotations

import os
import unittest

import numpy as np

try:
    import superdex.physics as physics
    from superdex.physics.utils.penetration import PenetrationChecker
    from superdex.physics.utils.scene_helpers import find_actor
except ImportError as error:  # pragma: no cover - the native build is absent
    if os.environ.get("SUPERDEX_REQUIRE_NATIVE"):
        raise
    raise unittest.SkipTest(f"penetration checker tests need the native build: {error}") from error

from .diffsim import scenes

DT = 0.01
NUM_STEPS = 40


def setUpModule() -> None:
    physics.initialize(num_worker_threads=0)


def tearDownModule() -> None:
    physics.shutdown()


class PenetrationCheckerTest(unittest.TestCase):
    def _run(self, scene, checker: PenetrationChecker, num_steps: int = NUM_STEPS) -> None:
        for step in range(num_steps):
            scene.step(DT)
            checker.record(step)

    @staticmethod
    def _resting_cube(penalty: float):
        """A cube resting on the ground with the given contact stiffness."""
        contact = physics.ContactParams(penalty_coefficient=penalty, coulomb_friction_coefficient=0.5)
        scene = physics.create_scene(f"resting_cube_{penalty:g}")
        scene.set_gravity([0.0, 0.0, -9.81])
        scene.create_rigid_actor(
            name="ground",
            shape=physics.create_plane_shape(normal=[0.0, 0.0, 1.0], distance=0.0),
            is_static=True,
            contact=contact,
        )
        half = float(np.abs(scenes.CUBE_COORDS).max())
        cube = scene.create_rigid_actor(
            name="cube",
            shape=scenes.cube_shape(),
            density=1000.0,
            contact=contact,
            world_from_local=physics.TransformRT([0.0, 0.0, half]),
        )
        return scene, cube, contact

    def test_resting_cube_overlaps_the_ground(self) -> None:
        """A dynamic cube at rest on a static plane: the deepest sample lies inside the
        ground, by less than the linear compression weight / (stiffness * face area) plus
        the penalty's smoothing distance (the force ramps up over it), by less than that
        smoothing distance alone at the engine's default stiffness, and by ten times less
        at the stiffer of two contacts; the only recorded pair is (cube, ground)."""
        depths = {}
        for penalty in (1e6, 1e9):
            scene, cube, contact = self._resting_cube(penalty)
            self.addCleanup(physics.destroy_scene, scene)
            checker = PenetrationChecker(scene)
            self.assertEqual([a.get_name() for a in checker.actors], [cube.get_name()])
            with self.assertRaises(physics.Error):
                checker.record()  # the query has no data before the first step
            self._run(scene, checker)
            pairs = checker.worst()
            self.assertEqual([p.names for p in pairs], [("cube", "ground")])
            depths[penalty] = pairs[0].depth
            half = float(np.abs(scenes.CUBE_COORDS).max())
            linear = cube.get_mass() * 9.81 / (penalty * (2.0 * half) ** 2)
            self.assertGreater(pairs[0].depth, 0.0)
            self.assertLess(
                pairs[0].depth, linear + 2.0 * contact.penalty_smoothing_half_distance, (penalty, linear)
            )
            self.assertGreater(pairs[0].num_contacts, 0)
        self.assertLess(depths[1e9], physics.ContactParams().penalty_smoothing_half_distance, depths)
        self.assertLess(depths[1e9], 0.1 * depths[1e6], depths)
        depth = depths[1e9]
        pairs = checker.worst()
        self.assertLess(abs(pairs[0].position[2]), 2.0 * depth + 1e-6, "the deepest sample sits at the ground plane")
        self.assertIn(f"{1000.0 * depth:.2f} mm", checker.report())
        with self.assertRaisesRegex(RuntimeError, "interpenetration above"):
            checker.assert_below(0.5 * depth)
        checker.assert_below(2.0 * depth)
        self.assertEqual(checker.max_depth(), depth)
        checker.reset()
        self.assertEqual(checker.worst(), [])
        self.assertEqual(checker.num_records, 0)

    def test_free_cube_has_no_contact(self) -> None:
        scene, cube = scenes.rigid_free()
        self.addCleanup(physics.destroy_scene, scene)
        checker = PenetrationChecker(scene)
        self._run(scene, checker, num_steps=5)
        self.assertEqual(checker.worst(), [])
        self.assertEqual(checker.max_depth(), 0.0)
        self.assertEqual(checker.report(), "no contact in 5 records")
        checker.assert_below(0.0)

    def test_two_cubes_report_both_pairs(self) -> None:
        scene, lower = scenes.two_cubes_on_plane("coulomb")
        self.addCleanup(physics.destroy_scene, scene)
        upper = find_actor(scene, "top")
        checker = PenetrationChecker(scene)
        self._run(scene, checker)
        names = {p.names for p in checker.worst()}
        lo, up = lower.get_name(), upper.get_name()
        self.assertIn(tuple(sorted((lo, up))), names)
        self.assertIn(tuple(sorted((lo, "ground"))), names)
        # The pair filter keeps only the cube-cube pair.
        filtered = PenetrationChecker(
            scene, actors=[upper], pair_names=lambda a, b: "ground" not in (a, b)
        )
        for step in range(5):
            scene.step(DT)
            filtered.record(step)
        self.assertEqual({p.names for p in filtered.worst()}, {tuple(sorted((lo, up)))})
        for pair in checker.worst():
            self.assertLess(pair.depth, 0.005, pair)

    def test_articulated_link_pushing_a_cube(self) -> None:
        """The controlled chain's lower link pushes the cube: the checker queries the
        chain's nested link actors (the articulated actor itself carries no contact
        samples) and a pair naming that link and the cube appears once they touch."""
        scene, chain, cube = scenes.chain_pushing_cube()
        self.addCleanup(physics.destroy_scene, scene)
        checker = PenetrationChecker(scene)
        self.assertEqual(
            {a.get_name() for a in checker.actors},
            {f"{chain.get_name()}/l0", f"{chain.get_name()}/l1", cube.get_name()},
        )
        targets = np.stack([np.linspace(0.0, -1.2, 30), np.zeros(30)], axis=1)
        for step in range(30):
            chain.set_articulated_target_pose(np.ascontiguousarray(targets[step]))
            scene.step(DT)
            checker.record(step)
        pairs = {p.names: p for p in checker.worst()}
        contact_pairs = [names for names in pairs if cube.get_name() in names and "ground" not in names]
        self.assertEqual(len(contact_pairs), 1, pairs.keys())
        other = [n for n in contact_pairs[0] if n != cube.get_name()][0]
        self.assertEqual(other, f"{chain.get_name()}/l1")
        self.assertGreater(pairs[contact_pairs[0]].depth, 0.0)
        self.assertLess(pairs[contact_pairs[0]].depth, 0.01)

    def test_query_leaves_the_adjoint_unchanged(self) -> None:
        """The contact-point query must not change the differentiable rollout: the engine
        stores the forward contact forces in the container that back-propagation reuses for
        the force adjoints whenever a contact query is registered, so the adjoint must zero
        it for every contact query (it once did so only for the total-force query, and the
        checker's query turned the gradient of a chain pushing a cube into 1e10)."""
        from superdex.physics.diffsim_rollout import DifferentiableRollout

        from .diffsim.harness import TranslationErrorLoss, configure_for_differentiability

        if not physics.uses_double_precision():
            raise unittest.SkipTest("the adjoint's finite-difference self-check needs double precision")
        num_steps = 30
        controls = np.stack([np.linspace(0.0, -1.2, num_steps), np.zeros(num_steps)], axis=1)

        def gradient(with_query: bool):
            scene, chain, cube = scenes.chain_pushing_cube()
            self.addCleanup(physics.destroy_scene, scene)
            configure_for_differentiability(scene)
            checker = PenetrationChecker(scene) if with_query else None
            start = np.asarray(cube.get_center_of_mass_transform().translation, dtype=np.float64)
            loss = TranslationErrorLoss(cube, ref=start + np.array([0.08, 0.0, 0.0]))
            result = DifferentiableRollout(scene, dt=DT, num_steps=num_steps).run(
                apply_inputs=lambda step: chain.set_articulated_target_pose(
                    np.ascontiguousarray(controls[step])
                ),
                terminal_losses=[loss],
            )
            self.assertTrue(result.fd_valid, result.flagged_steps)
            if checker is not None:
                checker.record()
                self.assertGreater(checker.max_depth(), 0.0, "the query must still see the contact")
            return result.loss, result.gradients[chain.get_name()].control_targets.copy(), result.max_adjoint_residual

        loss_plain, grad_plain, residual_plain = gradient(False)
        loss_query, grad_query, residual_query = gradient(True)
        self.assertAlmostEqual(loss_query, loss_plain, delta=1e-12)
        self.assertLess(residual_query, 1e-4, residual_query)
        rel = np.linalg.norm(grad_query - grad_plain) / np.linalg.norm(grad_plain)
        self.assertLess(rel, 1e-6, (rel, np.linalg.norm(grad_plain), np.linalg.norm(grad_query)))

    def test_query_survives_a_state_restore(self) -> None:
        scene, cube = scenes.rigid_on_plane("none", initial_velocity=(0.0, 0.0, 0.0))
        self.addCleanup(physics.destroy_scene, scene)
        checker = PenetrationChecker(scene)
        state = scene.capture_state()
        self.addCleanup(scene.release_all_states)
        self._run(scene, checker, num_steps=10)
        first = checker.max_depth()
        scene.restore_state(state, False)
        checker.reset()
        self._run(scene, checker, num_steps=10)
        self.assertAlmostEqual(checker.max_depth(), first, delta=1e-12)


if __name__ == "__main__":
    unittest.main()
