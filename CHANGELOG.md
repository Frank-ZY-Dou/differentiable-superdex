# Changelog – Project SuperDex

All notable changes to this repository will be documented here.

## [1.0.0+diffsim.1] - 2026-09-07

The differentiable-simulation fork of SuperDex 1.0.0. `+diffsim.N` counts this fork's releases on
that base; its wheels are built from this repository, not from PyPI. Includes the upstream `main`
branch as of 2026-09-06.

### Added

- Differentiable simulation. `superdex.physics.diffsim` is the per-step adjoint API;
  `superdex.physics.diffsim_rollout.DifferentiableRollout` is the rollout driver (per-step
  checkpoints, terminal and running losses, truncated sweeps, gradient clipping, solver
  diagnostics, failure-adaptive substepping); `superdex.physics.diffsim_torch` provides
  `TorchRollout`, an autograd bridge with controls, external forces, gravity, contact materials,
  densities, soft initial states and soft material parameters as tensors, and `PolicyRollout`,
  which trains a torch policy in the loop from joint-pose, position, orientation, soft-centroid
  and contact-force observations.
- Coverage: rigid, articulated (with pose controllers), soft and rod actors; contact between them
  and against static colliders, including soft-body SDF colliders and triangle-mesh colliders;
  node-to-rigid constraints. Losses on positions, orientations, joint poses, soft displacements
  and contact forces. Gradients for initial states, per-step controller targets and external
  forces, gravity, the contact material of either owner of a pair, densities and soft material
  parameters.
- Forward robustness for differentiable scenes: friction continuation
  (`NonLinearSolverParams.friction_continuation_levels`) and failure-adaptive substepping in the
  rollout driver.
- `superdex.physics.utils.penetration.PenetrationChecker`: the interpenetration of a scene
  measured from the engine's contact samples, with a report and an assertion.
- Examples in `superdex_physics/examples`: `example_diffsim_throw.py` (the per-step API by hand),
  `example_diffsim_video.py` (a rigid throw and a soft landing), `example_diffsim_sysid.py`
  (friction and density, or soft material parameters, identified from trajectories) and
  `example_diffsim_robot_video.py` (ten manipulation tasks on video, with finite-difference
  checks).
- Tests: 174 gradient checks against central finite differences and closed-form references in
  `superdex_physics/wheels/superdex-physics/test/diffsim`, run in both precisions by the
  `diffsim-tests` workflow, plus C++ unit tests of the engine-side pieces.

### Engine fixes for the adjoint

All of these act in differentiable scenes only (`make_scene_differentiable`); ordinary scenes
produce trajectories identical to upstream SuperDex.

- Residual sizing across assemblies; the inner solver's convergence norm and a relative outer
  tolerance with the true residual reported; stage-start contact Jacobians for deformable-versus-
  dynamic contact; exact adjoints across steps of different sizes and `get_step_jacobian` for
  changing step sizes; the moving-chart term of external torques (torque gradients now exact to
  1e-8); the SDF Hessian and collider-rotation terms of the contact-force adjoint; the direct
  dependence of contact-force losses on the contact parameters; in-domain finite-difference steps
  for small positive contact coefficients; contact onset on soft-body SDF colliders; queries
  refreshed after a state restore.

### Driver fixes

- Captured states released on every failure path; the truncation window no longer changes the
  objective; loss factories called once per step; actors resolved by identity, with actors of
  another scene and repeated actors rejected at construction; the torch bridges refuse the
  single-precision engine.

### Upstream

- Merged `facebookresearch/project_superdex` `main` through 2026-09-06: wheel projects under their
  components, `fp32`/`fp64` names, contact pair overrides, solver termination classes, sphere-tree
  contact culling, integration bundles.

## [2026-08-24]

- Initial release.
