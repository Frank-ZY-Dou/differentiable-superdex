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

"""Gradient-consistency tests for ``superdex.physics.diffsim`` that run from the
open-source export.

The engine's own differentiability tests (``mochi_physics/private/diffsim/test``)
are gated on ``MOCHI_INTERNAL`` because their ``.mochi_scene`` assets are not
shipped. This package rebuilds equivalent scenes programmatically (no assets)
and ports the same forward/reverse/finite-difference recipe to Python.

Run with the double-precision payload installed::

    cd superdex_physics/wheels/superdex-physics
    SUPERDEX_PRECISION=double python -m unittest test.diffsim.test_diffsim_gradients -v
"""
