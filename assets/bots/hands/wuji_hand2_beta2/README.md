# Wuji Hand 2 (Beta 2) Description

## Overview

This package contains a robot description of the Wuji Hand 2 (Beta 2) for SuperDex Physics, converted from the URDF and STL assets of the upstream [wuji-description](https://github.com/wuji-technology/wuji-description) repository (`hand2/hand2_beta2/body`, package version 2026.8.19, commit `4c1073d`). The Beta 1 model shipped with SuperDex is in [`../wuji_hand2_beta1`](../wuji_hand2_beta1).

Conventions of the wrist frame: the fingers extend along -z, they curl toward +y (the palm side), the thumb rests on the +x side of the right hand. Each hand has 20 revolute joints with the same names as the Beta 1 model (`*_mcp_flex`, `*_mcp_abd`, `*_pip`, `*_dip` per finger, `r_thumb_cmc_flex`, `r_thumb_cmc_abd`, `r_thumb_mcp`, `r_thumb_ip`), plus fixed tip sensor frames.

## Modifications

The package was produced by `tools/urdf_to_superdex_bot.py` of this repository:

- Kinematics, inertias and joint limits are taken from the URDF through SuperDex's URDF importer.
- Collision meshes: the upstream meshes where they are closed and small; the convex hull of the mesh otherwise (the engine bakes a signed distance field at bot creation and needs a closed surface).
- Visual meshes converted to GLB (Y-up), colored with the URDF materials.
- Contact between the wrist and the links not attached to it is disabled.

## License

The upstream assets are provided under the [MIT License](https://github.com/wuji-technology/wuji-description/blob/main/LICENSE) (Copyright (c) 2025 Wuji Technology); see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

**You are responsible for ensuring your use is compatible with all third-party licenses.**
