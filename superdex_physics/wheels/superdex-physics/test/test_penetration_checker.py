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
cube names the link, and a state restore keeps the query alive.

``PenetrationCheckerContractTest`` pins the checker's contract: a soft actor without a
collider (the engine's default for soft actors) is watched by default, a contact seen from
two watched bodies is counted once, two actors sharing a name stay apart, and no unobserved
or invalid state - no record, a non-finite sample, a non-finite or negative limit - gets a
safe verdict. The sample validation is fed synthetic rows through the checker's
``_contact_points`` seam (the engine emits no non-finite samples in a healthy run)."""

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


class _Row:
    """A synthetic contact-point row (the fields the checker reads)."""

    def __init__(self, actor_a, actor_b, distance: float, position, sample_index: int):
        self.actor_a = actor_a
        self.actor_b = actor_b
        self.distance = distance
        self.pos_a = np.asarray(position, dtype=np.float64)
        self.sample_index = sample_index


class _FedChecker(PenetrationChecker):
    """A checker reading the rows of ``self.rows`` instead of the engine's query."""

    rows: list = []

    def _contact_points(self, actor):
        return self.rows


class PenetrationCheckerContractTest(unittest.TestCase):
    def _run(self, scene, checker: PenetrationChecker, num_steps: int = NUM_STEPS) -> None:
        for step in range(num_steps):
            scene.step(DT)
            checker.record(step)

    def test_soft_actor_without_collider_is_watched_by_default(self) -> None:
        """A soft actor created from Python carries no collider yet emits contact samples
        against the ground's collider: the default selection watches it (it required a
        collider until 2026-09-10 and reported "no contact" for the jelly on the plane),
        and an explicit selection of the same actor observes the same depth."""
        scene, jelly = scenes.soft_cube_on_plane("rich")
        self.addCleanup(physics.destroy_scene, scene)
        self.assertEqual(jelly.get_collider_type(), physics.ColliderType.NONE)
        checker = PenetrationChecker(scene)
        self.assertEqual([a.get_handle().value for a in checker.actors], [jelly.get_handle().value])
        self._run(scene, checker, num_steps=10)
        pairs = checker.worst()
        self.assertEqual([p.names for p in pairs], [("ground", "jelly")])
        self.assertGreater(pairs[0].depth, 0.0)
        self.assertLess(pairs[0].depth, 0.01, pairs[0])
        self.assertGreater(pairs[0].num_contacts, 0)
        scene2, jelly2 = scenes.soft_cube_on_plane("rich")
        self.addCleanup(physics.destroy_scene, scene2)
        explicit = PenetrationChecker(scene2, actors=[jelly2])
        self._run(scene2, explicit, num_steps=10)
        self.assertAlmostEqual(explicit.max_depth(), pairs[0].depth, delta=1e-12)

    def test_contact_seen_from_both_bodies_is_counted_once(self) -> None:
        """Two stacked cubes, both watched: each cube's query lists the samples of the
        other cube against its collider too (the same sample twice over the two queries).
        The pair's sample count is the distinct samples, the same count a checker watching
        only the lower cube reports, and the raw rows are twice that."""
        scene, lower = scenes.two_cubes_on_plane("coulomb")
        self.addCleanup(physics.destroy_scene, scene)
        upper = find_actor(scene, "top")
        both = PenetrationChecker(scene)
        self.assertEqual(len(both.actors), 2)
        self._run(scene, both)
        both.reset()
        both.record()  # one record: the last step's rows
        key = tuple(sorted((lower.get_name(), upper.get_name())))
        pair = {p.names: p for p in both.worst()}[key]
        handles = {lower.get_handle().value, upper.get_handle().value}
        raw = sum(
            1
            for actor in (lower, upper)
            for point in actor.get_contact_points_world()
            if {point.actor_a.value, point.actor_b.value} == handles
        )
        self.assertGreater(pair.num_contacts, 0)
        self.assertEqual(raw, 2 * pair.num_contacts, (raw, pair))
        scene2, lower2 = scenes.two_cubes_on_plane("coulomb")
        self.addCleanup(physics.destroy_scene, scene2)
        one = PenetrationChecker(scene2, actors=[lower2])
        self._run(scene2, one)
        one.reset()
        one.record()
        pair_one = {p.names: p for p in one.worst()}[key]
        self.assertEqual(pair_one.num_contacts, pair.num_contacts)
        self.assertAlmostEqual(pair_one.depth, pair.depth, delta=1e-12)

    def test_same_named_actors_stay_apart(self) -> None:
        """Two dynamic cubes both named "cube" resting on the ground: the checker keeps
        one pair per handle pair (two "cube / ground" pairs, different handles) while a
        record's name-keyed summary holds the deeper of the two."""
        contact = physics.ContactParams(penalty_coefficient=1e6, coulomb_friction_coefficient=0.5)
        scene = physics.create_scene("same_names")
        self.addCleanup(physics.destroy_scene, scene)
        scene.set_gravity([0.0, 0.0, -9.81])
        scene.create_rigid_actor(
            name="ground",
            shape=physics.create_plane_shape(normal=[0.0, 0.0, 1.0], distance=0.0),
            is_static=True,
            contact=contact,
        )
        half = float(np.abs(scenes.CUBE_COORDS).max())
        cubes = [
            scene.create_rigid_actor(
                name="cube",
                shape=scenes.cube_shape(),
                density=density,
                contact=contact,
                world_from_local=physics.TransformRT([x, 0.0, half]),
            )
            for x, density in ((0.0, 1000.0), (0.5, 4000.0))
        ]
        checker = PenetrationChecker(scene)
        self.assertEqual(len(checker.actors), 2)
        summary = None
        for step in range(NUM_STEPS):
            scene.step(DT)
            summary = checker.record(step)
        pairs = checker.worst()
        self.assertEqual([p.names for p in pairs], [("cube", "ground"), ("cube", "ground")])
        self.assertNotEqual(pairs[0].handles, pairs[1].handles)
        self.assertEqual(
            {p.handles for p in pairs},
            {tuple(sorted((c.get_handle().value, find_actor(scene, "ground").get_handle().value))) for c in cubes},
        )
        # The denser cube sinks deeper; the name-keyed summary keeps the deeper value.
        self.assertGreater(pairs[0].depth, pairs[1].depth)
        self.assertEqual(summary[("cube", "ground")], max(checker.record()[("cube", "ground")], 0.0))

    def _fed(self):
        scene, cube = scenes.rigid_on_plane("none", initial_velocity=(0.0, 0.0, 0.0))
        self.addCleanup(physics.destroy_scene, scene)
        ground = find_actor(scene, "ground")
        checker = _FedChecker(scene)
        return checker, cube.get_handle(), ground.get_handle()

    def test_unobserved_scene_gets_no_safe_verdict(self) -> None:
        """No record is not "no contact": the verdict and the maximum raise, the report
        says so, and a checker without any actor to watch is rejected."""
        checker, cube, ground = self._fed()
        self.assertEqual(checker.num_records, 0)
        self.assertIn("no contact record yet", checker.report())
        with self.assertRaisesRegex(RuntimeError, "no contact record"):
            checker.assert_below(0.01)
        with self.assertRaisesRegex(RuntimeError, "no contact record"):
            checker.max_depth()
        checker.rows = []
        checker.record(0)
        self.assertEqual(checker.report(), "no contact in 1 records")
        self.assertEqual(checker.max_depth(), 0.0)
        checker.assert_below(0.0)
        with self.assertRaisesRegex(ValueError, "no actor to watch"):
            PenetrationChecker(checker.scene, actors=[])
        static_only = physics.create_scene("static_only")
        self.addCleanup(physics.destroy_scene, static_only)
        static_only.create_rigid_actor(
            name="ground",
            shape=physics.create_plane_shape(normal=[0.0, 0.0, 1.0], distance=0.0),
            is_static=True,
        )
        with self.assertRaisesRegex(ValueError, "no actor to watch"):
            PenetrationChecker(static_only)
        with self.assertRaisesRegex(ValueError, "listed twice"):
            PenetrationChecker(checker.scene, actors=[checker.actors[0], checker.actors[0]])

    def test_invalid_limit_is_rejected(self) -> None:
        checker, cube, ground = self._fed()
        checker.rows = [_Row(cube, ground, -0.001, (0.0, 0.0, -0.001), 0)]
        checker.record(0)
        for limit in (float("nan"), float("inf"), -float("inf"), -1e-3):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    checker.assert_below(limit)
                with self.assertRaises(ValueError):
                    checker.report(limit)
        checker.assert_below(0.002)
        with self.assertRaisesRegex(RuntimeError, "interpenetration above"):
            checker.assert_below(0.0005)

    def test_non_finite_samples_fail_closed(self) -> None:
        """A non-finite distance or position raises at record time in either row order,
        nothing of that record is kept, and no safe verdict follows until a reset."""
        checker, cube, ground = self._fed()
        shallow = _Row(cube, ground, -0.001, (0.0, 0.0, -0.001), 0)
        deep = _Row(cube, ground, -0.02, (0.1, 0.0, -0.02), 1)
        checker.rows = [shallow]
        checker.record(0)
        bad_rows = {
            "nan distance": _Row(cube, ground, float("nan"), (0.0, 0.0, 0.0), 2),
            "inf distance": _Row(cube, ground, -float("inf"), (0.0, 0.0, 0.0), 2),
            "nan position": _Row(cube, ground, -0.003, (0.0, float("nan"), 0.0), 2),
            "inf position": _Row(cube, ground, -0.003, (float("inf"), 0.0, 0.0), 2),
        }
        for label, bad in bad_rows.items():
            for order in ((bad, deep), (deep, bad)):
                with self.subTest(label=label, first=order[0] is bad):
                    checker.rows = list(order)
                    with self.assertRaisesRegex(ValueError, "invalid contact sample"):
                        checker.record(1)
                    self.assertEqual(checker.num_records, 1)
                    self.assertEqual([p.depth for p in checker.worst()], [0.001], "the deep row was not folded")
                    self.assertIn("rejected for non-finite samples", checker.report())
                    with self.assertRaisesRegex(RuntimeError, "rejected for non-finite"):
                        checker.assert_below(0.05)
                    with self.assertRaisesRegex(RuntimeError, "rejected for non-finite"):
                        checker.max_depth()
        self.assertEqual(checker.num_invalid_records, 8)
        checker.reset()
        self.assertEqual(checker.num_invalid_records, 0)
        checker.rows = [deep, shallow]
        checker.record(2)
        self.assertAlmostEqual(checker.max_depth(), 0.02)
        with self.assertRaisesRegex(RuntimeError, "20.00 mm at step 2"):
            checker.assert_below(0.01)
        # A depth mutated to a non-finite value after the record is caught too.
        checker.worst()[0].depth = float("nan")
        with self.assertRaisesRegex(RuntimeError, "invalid stored depth"):
            checker.assert_below(1.0)

    def test_pair_filter_sees_names_in_name_order(self) -> None:
        """The ``pair_names`` predicate receives the two names in name order whatever the
        handle order (the ground here has the lower handle but the later name): a filter
        written for the report's order keeps selecting the pair."""
        checker, cube, ground = self._fed()
        self.assertLess(ground.value, cube.value)
        self.assertLess("cube", "ground")
        seen: list[tuple[str, str]] = []

        def only_cube_ground(a: str, b: str) -> bool:
            seen.append((a, b))
            return (a, b) == ("cube", "ground")

        filtered = _FedChecker(checker.scene, actors=[checker.actors[0]], pair_names=only_cube_ground)
        filtered.rows = [_Row(cube, ground, -0.01, (0.0, 0.0, -0.01), 0), _Row(ground, cube, -0.002, (0.0, 0.0, 0.0), 0)]
        filtered.record(0)
        self.assertEqual(seen, [("cube", "ground"), ("cube", "ground")])
        self.assertEqual([(p.names, p.depth) for p in filtered.worst()], [(("cube", "ground"), 0.01)])

    def test_duplicate_rows_are_validated_before_they_are_dropped(self) -> None:
        """A second copy of a sample (the other body's query) with a non-finite datum is
        rejected even though the first copy was valid, and so is a non-finite row of a pair
        the filter excludes."""
        checker, cube, ground = self._fed()
        good = _Row(cube, ground, -0.001, (0.0, 0.0, -0.001), 0)
        checker.rows = [good, _Row(cube, ground, float("nan"), (0.0, 0.0, -0.001), 0)]
        with self.assertRaisesRegex(ValueError, "invalid contact sample"):
            checker.record(0)
        self.assertEqual((checker.num_records, checker.num_invalid_records), (0, 1))
        excluded = _FedChecker(checker.scene, actors=[checker.actors[0]], pair_names=lambda a, b: False)
        excluded.rows = [_Row(cube, ground, -0.001, (float("nan"), 0.0, 0.0), 1)]
        with self.assertRaisesRegex(ValueError, "invalid contact sample"):
            excluded.record(0)

    def test_sample_identity_is_the_emitting_actor_and_index(self) -> None:
        """Fed rows: the same sample listed twice (as the two bodies' queries do) counts
        once; a sample of the cube on the ground and one of the ground on the cube with
        the same index are two samples; the pair depth is the deepest of both directions
        whatever the order."""
        checker, cube, ground = self._fed()
        on_ground = _Row(cube, ground, -0.001, (0.0, 0.0, -0.001), 3)
        on_cube = _Row(ground, cube, -0.004, (0.0, 0.0, 0.0), 3)
        checker.rows = [on_ground, on_ground]
        depths = checker.record(0)
        pair = checker.worst()[0]
        self.assertEqual((pair.num_contacts, pair.depth), (1, 0.001))
        self.assertEqual(depths, {("cube", "ground"): 0.001})
        for order in ((on_ground, on_cube), (on_cube, on_ground, on_ground)):
            checker.reset()
            checker.rows = list(order)
            checker.record(1)
            pair = checker.worst()[0]
            self.assertEqual(pair.num_contacts, 2, order)
            self.assertEqual(pair.depth, 0.004)
            self.assertEqual(pair.names, ("cube", "ground"))
            self.assertEqual(pair.handles, tuple(sorted((cube.value, ground.value))))
            np.testing.assert_array_equal(pair.position, on_cube.pos_a)


if __name__ == "__main__":
    unittest.main()
