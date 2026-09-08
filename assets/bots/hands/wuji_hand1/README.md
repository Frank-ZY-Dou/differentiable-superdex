# Wuji Hand 1 Description

## Overview

This package contains a robot description of the Wuji Hand 1 (v1.0.2) for SuperDex Physics, converted from the URDF and STL assets of the upstream [wuji_hand_description](https://github.com/wuji-technology/wuji_hand_description) repository (`urdf/right.urdf`, `urdf/left.urdf`, `meshes/`).

Conventions of the palm frame: the fingers extend along +z, they curl toward +x (the palm side), the thumb (`finger1`) rests on the +y side of the right hand. Each hand has 20 revolute joints: four per finger (`fingerN_joint1` and `joint3`, `joint4` flex, `joint2` abducts; for the thumb `joint1` and `joint2` are the carpometacarpal pair).

## Modifications

The package was produced by `tools/urdf_to_superdex_bot.py` of this repository:

- Kinematics, inertias and joint limits are taken from the URDF through SuperDex's URDF importer.
- Collision meshes: the upstream `*_collision.STL` meshes where they are closed; the convex hull of the mesh where it is not (the engine bakes a signed distance field at bot creation and needs a closed surface).
- Visual meshes converted to GLB (Y-up), colored with the URDF materials.
- Contact between the palm and the links not attached to it is disabled.

## License

The upstream assets are provided under the [MIT License](https://github.com/wuji-technology/wuji_hand_description/blob/main/LICENSE) (Copyright (c) 2025 Wuji Technology); see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

**You are responsible for ensuring your use is compatible with all third-party licenses.**
