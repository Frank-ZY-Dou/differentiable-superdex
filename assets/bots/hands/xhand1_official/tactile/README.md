# XHand1 taxel layout

Three files copied verbatim from https://github.com/tsingqingyun/xhand1 (`tactile/`,
commit 50fb891cd8a5aa71959c831977637b8a9279e054, 2026-07-10), a third-party MuJoCo
conversion of RobotEra's XHand1 URDF v1.3 delivery with tactile data, and are reproduced
here with thanks to that repository, which is the source of the layout.

| file | sensor | attached link (`{side}` = left / right) | sha256 (first 16) |
|---|---|---|---|
| `t30_right.json` | T30, right thumb | `{side}_hand_thumb_rota_link2` | d816237de7543a1f |
| `t30_left.json`  | T30, left thumb  | `{side}_hand_thumb_rota_link2` | 3faad15a4845438c |
| `t16.json`       | T16, index / mid / ring / pinky (shared) | `{side}_hand_{index_rota,mid,ring,pinky}_link2` | fc58f8da95f0a1ab |

Each file lists 120 `measurement_points` (`point` 1..120, `x y z` in **millimetres**),
already expressed in the distal link's URDF frame; the `coordinate_system` string
describes the transform that was applied, do not apply it again. Points are row-major
on a 10 x 12 grid: `point-1 = 12*row + col`, columns run along the finger (base ->
tip, ~2 mm pitch), rows wrap around the pad. The left and right T30 files differ by a
0.78 mm z offset, so they are loaded separately and the left thumb is not a mirror of
the right one.

Registered against the meshes in `../meshes` (identical to the ones in that repo):
all 600 points lie within 1.2 mm of the STL surface (four fingers 0.27 mm outside,
right thumb 0.16 mm outside, left thumb 0.21 mm inside).
