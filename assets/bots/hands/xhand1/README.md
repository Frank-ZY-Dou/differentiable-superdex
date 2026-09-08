# XHand1 Description

## Overview

This package contains a robot description of the RobotEra XHand1 for SuperDex Physics, converted from the URDF and STL assets redistributed by [DexMachina](https://github.com/MandiZhao/dexmachina) (`dexmachina/assets/xhand`, `xhand_right.urdf`, `xhand_left.urdf`, `meshes/`).

Conventions of the root frame (`right_hand_link`, which includes the wrist mount): the fingers extend along -z, they curl toward +y (the palm side), the thumb rests on the +x side of the right hand. Each hand has 12 revolute joints: two flexion joints per finger (`*_joint1`, `*_joint2`), an abduction joint of the index (`*_index_bend_joint`) and three thumb joints (`*_thumb_bend_joint`, `*_thumb_rota_joint1`, `*_thumb_rota_joint2`).

## Modifications

The package was produced by `tools/urdf_to_superdex_bot.py` of this repository:

- Kinematics, inertias and joint limits are taken from the URDF through SuperDex's URDF importer.
- Collision meshes: the convex hull of each link's mesh (the upstream meshes are visual meshes, several of them open; the engine bakes a signed distance field at bot creation and needs a closed surface), the small tip markers kept as they are.
- Visual meshes converted to GLB (Y-up), colored with the URDF materials.
- Contact between the root link and the links not attached to it is disabled.

## License

The DexMachina assets are provided under the [MIT License](https://github.com/MandiZhao/dexmachina/blob/main/LICENSE) (Copyright (c) 2025 Mandi Zhao); see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The XHand1 is a product of RobotEra.

**You are responsible for ensuring your use is compatible with all third-party licenses.**
