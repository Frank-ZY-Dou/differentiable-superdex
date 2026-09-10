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

"""Interpenetration checker: how deep the bodies of a scene enter each other.

A penalty contact is compliant, so bodies in contact always overlap a little; the
overlap grows with the load and shrinks with the contact stiffness
(:attr:`~superdex.physics.ContactParams.penalty_coefficient`). The checker reports it
from the engine's own contact samples: every sample the collision detection produces
carries the signed distance of the sampled surface point to the other body (negative
inside), so the deepest sample of an actor pair is the penetration the physics sees.
It is a sampled measure, not a collision certificate: what the samples do not see (a
vertex between two samples, a render model larger than the collision model, a thin
body crossing between two samples within one step) the checker does not see either.

Register the checker before the first step of the scene (the contact query must exist
before a step for its data to be available after it; reading it before any step is an
engine error), call :meth:`record` after every step - after every accepted substep of
a substepped rollout, or the peaks inside a split step go unseen - then :meth:`report`
or :meth:`assert_below`::

    checker = PenetrationChecker(scene)
    for step in range(num_steps):
        scene.step(dt)
        checker.record(step)
    print(checker.report())
    checker.assert_below(0.002)  # no pair deeper than 2 mm

Which actors are watched. An actor emits contact samples from its own surface against
the colliders of the other bodies whether or not it carries a collider itself (a
collider only lets the *other* bodies see it), so the default selection is by the
contact-point query's capability: every non-static rigid, soft, shell or rod actor of
the scene, the nested link actors of the articulations included (an articulated actor
carries no samples of its own, its links do). The static bodies they touch, e.g. the
ground, appear as the other member of a pair. A watched actor's query also lists the
samples of other bodies against its collider, so a contact seen from two watched bodies
is listed twice; the checker counts every sample once (identity: emitting actor, other
actor, sample index), and a sample of A on B and a sample of B on A are two samples.
Actors are told apart by handle; names only label the report.

Fail-closed rules. A non-finite distance or position in the contact data raises at
:meth:`record` time, leaves the running maxima untouched and is remembered: until
:meth:`reset` the checker gives no safe verdict. :meth:`assert_below` and
:meth:`max_depth` raise without a record (nothing was observed), and a non-finite or
negative limit is rejected. "No contact" is only ever reported for observed records.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np
import superdex.physics as physics

__all__ = ["PairPenetration", "PenetrationChecker"]


@dataclasses.dataclass
class PairPenetration:
    """The deepest contact sample of one actor pair over the recorded steps."""

    actor_a: str
    actor_b: str
    depth: float
    """[m] how far the deepest sample lies inside the other body (0 if none does)."""
    step: int | None
    """The ``step`` label of the record in which the deepest sample occurred."""
    position: np.ndarray
    """World position of that sample."""
    num_contacts: int
    """Distinct contact samples of the pair in that record, the samples of either body
    on the other counted separately and a sample listed by both bodies' queries once."""
    handles: tuple[int, int]
    """The two actors' handle values (the pair's identity; two actors may share a name)."""

    @property
    def names(self) -> tuple[str, str]:
        return (self.actor_a, self.actor_b)


class PenetrationChecker:
    """Records, per actor pair, the deepest contact sample after each step.

    Args:
        scene: the scene to watch.
        actors: the actors whose contact-point queries are read (default: every
            non-static actor of a type that carries contact samples - rigid, soft, shell,
            rod - with or without a collider of its own; see the module docstring). An
            actor of another type, or one the engine gives no contact samples, is
            rejected when its query is registered.
        pair_names: optional callable ``(actor_name_a, actor_name_b) -> bool`` selecting
            the pairs to record (default: all); it receives the two names in name order,
            as the report lists them. Rows it excludes are still validated.

    Raises:
        ValueError: no actor to watch, an actor listed twice, or an actor of a type
            without contact samples.
    """

    _QUERY_TYPES = (
        physics.ActorType.RIGID,
        physics.ActorType.SOFT,
        physics.ActorType.SHELL,
        physics.ActorType.ROD,
    )

    def __init__(self, scene, actors: Sequence | None = None, pair_names=None):
        self.scene = scene
        if actors is None:
            actors = []

            def select(actor) -> None:
                if not actor.is_static() and actor.get_type() in self._QUERY_TYPES:
                    actors.append(actor)

            scene.for_each_actor(select)
        self.actors = list(actors)
        if not self.actors:
            raise ValueError(
                "no actor to watch: the scene has no non-static actor with contact samples "
                "(pass actors=... to watch a specific set)"
            )
        watched: set[int] = set()
        for actor in self.actors:
            if actor.get_type() not in self._QUERY_TYPES:
                raise ValueError(
                    f"{actor.get_name()!r} ({actor.get_type()}) does not support the contact-point query"
                )
            handle = int(actor.get_handle().value)
            if handle in watched:
                raise ValueError(f"{actor.get_name()!r} is listed twice")
            watched.add(handle)
        self._queries = [actor.register_query(physics.QueryType.CONTACT_POINTS) for actor in self.actors]
        self._pair_filter = pair_names
        self._names: dict[int, str] = {}
        self._worst: dict[tuple[int, int], PairPenetration] = {}
        self.num_records = 0
        self.num_invalid_records = 0
        """Records rejected for a non-finite sample since the last :meth:`reset`."""

    def _name(self, handle) -> str:
        """The name of the actor behind ``handle`` (cached by handle value)."""
        value = int(handle.value)
        name = self._names.get(value)
        if name is None:
            actor = self.scene.get_actor(handle)
            name = actor.get_name() if actor is not None else f"<handle {value}>"
            self._names[value] = name
        return name

    @staticmethod
    def _contact_points(actor):
        """The rows of the actor's contact-point query (the seam the tests feed)."""
        return actor.get_contact_points_world()

    def record(self, step: int | None = None) -> dict[tuple[str, str], float]:
        """Reads the contact samples of the last step and folds them into the running
        maxima. Returns ``{(name_a, name_b): depth}`` for the pairs in contact in this
        record (depth 0 for pairs whose samples are all outside; pairs of same-named
        actors share a key here and keep the deeper value - :meth:`worst` tells them
        apart by handle).

        Raises:
            ValueError: a sample with a non-finite distance or position; nothing of
                the record is kept, and the rejection is counted in
                :attr:`num_invalid_records`.
        """
        current: dict[tuple[int, int], list] = {}
        seen: set[tuple[int, int, int]] = set()
        where = f" at step {step}" if step is not None else ""
        for actor in self.actors:
            for point in self._contact_points(actor):
                handle_a, handle_b = point.actor_a, point.actor_b
                sample = (int(handle_a.value), int(handle_b.value), int(point.sample_index))
                name_a, name_b = self._name(handle_a), self._name(handle_b)
                # Every row read is validated, before the duplicate and pair filters: a
                # non-finite datum is rejected wherever it appears.
                distance = float(point.distance)
                position = np.asarray(point.pos_a, dtype=np.float64).reshape(-1)
                if not (
                    np.isfinite(distance) and position.shape == (3,) and bool(np.all(np.isfinite(position)))
                ):
                    self.num_invalid_records += 1
                    raise ValueError(
                        f"invalid contact sample {sample[2]} of {name_a} on {name_b}{where}: "
                        f"distance {distance}, position {position.tolist()}"
                    )
                if sample in seen:
                    continue  # the same sample listed by the other body's query
                seen.add(sample)
                key = (sample[0], sample[1]) if sample[0] <= sample[1] else (sample[1], sample[0])
                # The pair filter sees the names in name order, as the report lists them.
                if self._pair_filter is not None and not self._pair_filter(*sorted((name_a, name_b))):
                    continue
                depth = -distance
                entry = current.get(key)
                if entry is None:
                    current[key] = [depth, position, 1]
                else:
                    entry[2] += 1
                    if depth > entry[0]:
                        entry[0] = depth
                        entry[1] = position
        result: dict[tuple[str, str], float] = {}
        for key, (depth, position, count) in current.items():
            depth = max(depth, 0.0)
            names = self._pair_names(key)
            result[names] = max(result.get(names, 0.0), depth)
            worst = self._worst.get(key)
            if worst is None or depth > worst.depth:
                self._worst[key] = PairPenetration(names[0], names[1], depth, step, position, count, key)
        self.num_records += 1
        return result

    def _pair_names(self, key: tuple[int, int]) -> tuple[str, str]:
        """The display names of a pair key, in name order."""
        a, b = self._names[key[0]], self._names[key[1]]
        return (a, b) if a <= b else (b, a)

    def reset(self) -> None:
        """Forgets the recorded maxima (the queries stay registered)."""
        self._worst.clear()
        self.num_records = 0
        self.num_invalid_records = 0

    def worst(self) -> list[PairPenetration]:
        """The recorded pairs, deepest first."""
        return sorted(self._worst.values(), key=lambda p: -p.depth)

    def _require_records(self) -> None:
        if self.num_invalid_records:
            raise RuntimeError(
                f"{self.num_invalid_records} records were rejected for non-finite contact samples: "
                "no safe verdict until reset()"
            )
        if self.num_records == 0:
            raise RuntimeError(
                f"no contact record: nothing was observed yet ({len(self.actors)} actors watched); "
                "call record() after every step"
            )

    def max_depth(self) -> float:
        """[m] the deepest sample over all pairs and records (0 without contact).

        Raises:
            RuntimeError: without a record (an unobserved scene is not a contact-free one).
        """
        self._require_records()
        return max((p.depth for p in self._worst.values()), default=0.0)

    @staticmethod
    def _check_limit(limit: float) -> float:
        limit = float(limit)
        if not np.isfinite(limit) or limit < 0.0:
            raise ValueError(f"the penetration limit must be a finite non-negative length, got {limit}")
        return limit

    def report(self, limit: float | None = None) -> str:
        """A table of the recorded pairs, deepest first; pairs deeper than ``limit`` [m]
        are flagged. Without a record the table says so instead of "no contact"."""
        if limit is not None:
            limit = self._check_limit(limit)
        rejected = (
            f" ({self.num_invalid_records} records rejected for non-finite samples)"
            if self.num_invalid_records
            else ""
        )
        if self.num_records == 0:
            return f"no contact record yet ({len(self.actors)} actors watched){rejected}"
        pairs = self.worst()
        if not pairs:
            return f"no contact in {self.num_records} records{rejected}"
        lines = [f"deepest contact sample per actor pair over {self.num_records} records{rejected}:"]
        for p in pairs:
            flag = "  EXCEEDS" if limit is not None and p.depth > limit else ""
            where = f" at step {p.step}" if p.step is not None else ""
            lines.append(
                f"  {p.actor_a} / {p.actor_b}: {1000.0 * p.depth:.2f} mm{where} "
                f"({p.num_contacts} samples){flag}"
            )
        return "\n".join(lines)

    def assert_below(self, limit: float) -> None:
        """Raises ``RuntimeError`` naming the pairs whose deepest sample exceeds ``limit``
        [m]. Also raises without a record (nothing was observed), after a record was
        rejected for non-finite samples, or with an invalid stored depth, and
        ``ValueError`` for a non-finite or negative limit: none of these is a safe
        verdict."""
        limit = self._check_limit(limit)
        self._require_records()
        invalid = [p for p in self._worst.values() if not np.isfinite(p.depth)]
        if invalid:
            raise RuntimeError(
                "invalid stored depth: " + "; ".join(f"{p.actor_a} / {p.actor_b} {p.depth}" for p in invalid)
            )
        offenders = [p for p in self.worst() if p.depth > limit]
        if offenders:
            raise RuntimeError(
                f"interpenetration above {1000.0 * limit:.2f} mm: "
                + "; ".join(
                    f"{p.actor_a} / {p.actor_b} {1000.0 * p.depth:.2f} mm"
                    + (f" at step {p.step}" if p.step is not None else "")
                    for p in offenders
                )
            )
