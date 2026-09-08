# Sharpa Wave Description

## Overview

This package contains a robot description of the Sharpa Wave hand for SuperDex Physics, converted from the URDF and STL assets of the upstream [sharpa-urdf-usd-xml](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml) repository (`wave_01/right_sharpa_wave/right_sharpa_wave.urdf`, `wave_01/left_sharpa_wave/left_sharpa_wave.urdf` and their `meshes/`, package version 1.0.0, commit `0d447b6`). The plain hand is converted, without the upstream flange and wrist variants; the fingertip elastomer links that carry the tactile sensor surfaces are kept, so tactile sensing can be added later.

Conventions of the root frame (`right_hand_C_MC`, the metacarpal body): the fingers extend along +z. Each hand has 22 revolute joints: the thumb's carpometacarpal and metacarpophalangeal pairs (`*_thumb_CMC_FE`, `*_thumb_CMC_AA`, `*_thumb_MCP_FE`, `*_thumb_MCP_AA`) and its interphalangeal joint (`*_thumb_IP`); flexion, abduction, proximal and distal joints of the index, middle and ring fingers (`*_MCP_FE`, `*_MCP_AA`, `*_PIP`, `*_DIP`); and the same four joints plus a metacarpal joint (`*_pinky_CMC`) for the pinky.

## Modifications

The package was produced by `tools/urdf_to_superdex_bot.py` of this repository:

- Kinematics, inertias and joint limits are taken from the URDF through SuperDex's URDF importer.
- Collision meshes: the upstream collision meshes where they are closed volumes of moderate size (faces without area dropped); the convex hull of the mesh otherwise (the engine bakes a signed distance field at bot creation and needs a closed surface).
- Visual meshes converted to GLB (Y-up), colored with the URDF materials.
- Contact between the root link and the links not attached to it is disabled.

## License

The upstream assets are provided under the Apache License, Version 2.0 (Copyright 2025 Sharpa Group); see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

**You are responsible for ensuring your use is compatible with all third-party licenses.**
