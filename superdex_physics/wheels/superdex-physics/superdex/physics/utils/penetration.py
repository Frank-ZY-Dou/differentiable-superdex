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
What the samples do not see (a vertex between two samples, a render model larger than
the collision model) the checker does not see either.

Register the checker before the first step of the scene (the contact query must exist
before a step for its data to be available after it; reading it before any step is an
engine error), call :meth:`record` after every step, then :meth:`report` or
:meth:`assert_below`::

    checker = PenetrationChecker(scene)
    for step in range(num_steps):
        scene.step(dt)
        checker.record(step)
    print(checker.report())
    checker.assert_below(0.002)  # no pair deeper than 2 mm
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
    """Contact samples of the pair in that record."""

    @property
    def names(self) -> tuple[str, str]:
        return (self.actor_a, self.actor_b)


class PenetrationChecker:
    """Records, per actor pair, the deepest contact sample after each step.

    Args:
        scene: the scene to watch.
        actors: the actors whose contacts are queried (default: every non-static rigid
            or soft actor of the scene with a collider, the nested link actors of the
            articulations included - an articulated actor carries no contact samples of
            its own, its links do; the static actors they touch, e.g. the ground, show up
            as the other member of a pair). A rod or shell actor is watched through the
            bodies it touches.
        pair_names: optional callable ``(actor_name_a, actor_name_b) -> bool`` selecting
            the pairs to record (default: all).
    """

    _QUERY_TYPES = (physics.ActorType.RIGID, physics.ActorType.SOFT)

    def __init__(self, scene, actors: Sequence | None = None, pair_names=None):
        self.scene = scene
        if actors is None:
            actors = []

            def select(actor) -> None:
                if (
                    not actor.is_static()
                    and actor.get_collider_type() != physics.ColliderType.NONE
                    and actor.get_type() in self._QUERY_TYPES
                ):
                    actors.append(actor)

            scene.for_each_actor(select)
        self.actors = list(actors)
        for actor in self.actors:
            if actor.get_type() not in self._QUERY_TYPES:
                raise ValueError(
                    f"{actor.get_name()!r} ({actor.get_type()}) does not support the contact-point query"
                )
        self._queries = [actor.register_query(physics.QueryType.CONTACT_POINTS) for actor in self.actors]
        self._pair_filter = pair_names
        self._names: dict = {}
        self._worst: dict[tuple[str, str], PairPenetration] = {}
        self.num_records = 0

    def _name(self, handle) -> str:
        name = self._names.get(handle)
        if name is None:
            actor = self.scene.get_actor(handle)
            name = actor.get_name() if actor is not None else f"<handle {handle.value}>"
            self._names[handle] = name
        return name

    def record(self, step: int | None = None) -> dict[tuple[str, str], float]:
        """Reads the contact samples of the last step. Returns ``{pair: depth}`` for the
        pairs in contact in this record (depth 0 for pairs whose samples are all outside)
        and folds them into the running maxima."""
        current: dict[tuple[str, str], list] = {}
        for actor in self.actors:
            for point in actor.get_contact_points_world():
                pair = self._pair_key(point.actor_a, point.actor_b)
                if self._pair_filter is not None and not self._pair_filter(*pair):
                    continue
                depth = -float(point.distance)
                entry = current.get(pair)
                if entry is None:
                    current[pair] = [depth, np.asarray(point.pos_a, dtype=np.float64), 1]
                else:
                    entry[2] += 1
                    if depth > entry[0]:
                        entry[0] = depth
                        entry[1] = np.asarray(point.pos_a, dtype=np.float64)
        result = {}
        for pair, (depth, position, count) in current.items():
            depth = max(depth, 0.0)
            result[pair] = depth
            worst = self._worst.get(pair)
            if worst is None or depth > worst.depth:
                self._worst[pair] = PairPenetration(pair[0], pair[1], depth, step, position, count)
        self.num_records += 1
        return result

    def _pair_key(self, handle_a, handle_b) -> tuple[str, str]:
        a, b = self._name(handle_a), self._name(handle_b)
        return (a, b) if a <= b else (b, a)

    def reset(self) -> None:
        """Forgets the recorded maxima (the queries stay registered)."""
        self._worst.clear()
        self.num_records = 0

    def worst(self) -> list[PairPenetration]:
        """The recorded pairs, deepest first."""
        return sorted(self._worst.values(), key=lambda p: -p.depth)

    def max_depth(self) -> float:
        """[m] the deepest sample over all pairs and records (0 without contact)."""
        return max((p.depth for p in self._worst.values()), default=0.0)

    def report(self, limit: float | None = None) -> str:
        """A table of the recorded pairs, deepest first; pairs deeper than ``limit`` are
        flagged."""
        pairs = self.worst()
        if not pairs:
            return f"no contact in {self.num_records} records"
        lines = [f"deepest contact sample per actor pair over {self.num_records} records:"]
        for p in pairs:
            flag = "  EXCEEDS" if limit is not None and p.depth > limit else ""
            where = f" at step {p.step}" if p.step is not None else ""
            lines.append(
                f"  {p.actor_a} / {p.actor_b}: {1000.0 * p.depth:.2f} mm{where} "
                f"({p.num_contacts} samples){flag}"
            )
        return "\n".join(lines)

    def assert_below(self, limit: float) -> None:
        """Raises ``RuntimeError`` naming the pairs whose deepest sample exceeds ``limit`` [m]."""
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
