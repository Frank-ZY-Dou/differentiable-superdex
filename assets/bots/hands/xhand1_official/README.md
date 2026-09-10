# XHand1 (official v1.3) with tactile taxel layouts

A SuperDex bot package for RobotEra's XHand1 dexterous hand, converted from the official
URDF v1.3 delivery, together with the taxel layouts of its fingertip tactile sensors. It is
used by the differentiable tactile example (`superdex_physics/examples/example_diffsim_tactile.py`).

This package is separate from `assets/bots/hands/xhand1`, which is converted from the DexMachina
copy and used by the five-finger grasp demo; the tactile taxel positions are registered against
the official v1.3 meshes, so the tactile example uses this package.

## Contents

- `right/`, `left/`: the bot packages (`xhand1_v13_{side}.superdex_bot` with baked collision and
  GLB render meshes), converted with `tools/urdf_to_superdex_bot.py`; the link frames match the URDF.
- `meshes/`: the URDF's visual STL meshes, per link. The tactile readout reads the five distal-link
  meshes to orient the taxel normals.
- `tactile/`: the 600 taxel measurement points (see `tactile/README.md` for the source and format).
- `xhand_{side}.urdf`, `CHANGELOG.md`: the upstream URDF v1.3 and its changelog.

## Attribution

The hand model is RobotEra's XHand1 (URDF v1.3 delivery). The taxel layouts are reproduced from
[tsingqingyun/xhand1](https://github.com/tsingqingyun/xhand1/tree/main/tactile) with thanks; see
`tactile/README.md`. Please credit those sources when you use this package, and check that your use
is compatible with RobotEra's terms for the XHand1 model.
