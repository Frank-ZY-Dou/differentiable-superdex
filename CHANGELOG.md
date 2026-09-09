# Changelog – Project SuperDex

All notable changes to this repository will be documented here.

## [Unreleased]

### Fixed

- `tools/urdf_to_superdex_bot.py`: a link without a collision mesh gets no shape, and the engine
  gives a shapeless link no mass; the tool now folds the inertial of such a link into its parent
  across a fixed joint (parallel-axis update in the parent's frame) and reports a shapeless link
  on a moving joint, which stays massless. A URDF whose root hangs on a fixed joint, or whose
  root is a bare `world` frame, now yields a fixed base (a Hard world joint) instead of a floating
  one; `--base fixed|floating` overrides the detection. The joint dynamics the importer reads
  from the URDF (viscous damping and Coulomb friction, joint inertia, limit stiffness and
  damping) are written to the package. The package verification moves every joint inside its
  limits instead of assuming a floating root, so fixed-base robots verify too. Mimic joints,
  which the importer does not honor, are reported.

### Added

- `test/diffsim/test_diffsim_rollout.py::ArticulatedInitialPoseTest`: the driver's initial-pose
  gradient of a torque-driven articulated actor against finite differences. The revolute
  pendulum agrees to about 2e-5; the new `scenes.chain_revolute_prismatic` (a prismatic joint
  after a revolute one) is off by about a tenth, an open issue of
  `set_articulated_pose_from_joints_backward` kept as an expected failure.

## [1.0.0+diffsim.1] - 2026-09-08

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
  (friction and density, or soft material parameters, identified from trajectories),
  `example_diffsim_ik.py` (inverse kinematics through the simulator: joint targets solved by
  L-BFGS on the settled pose, compensating the controller's sag under gravity, and with a
  contact-force objective against a box) and `example_diffsim_robot_video.py` (fifteen manipulation
  tasks on video, with finite-difference checks, among them grasps with six hands). With `--export-scenes` the demos also write the recorded frames' bodies, meshes,
  textures and camera, and `render_diffsim_blender.py` renders them with Blender.
- Assets: hand packages for the Wuji Hand 1, the Wuji Hand 2 (beta 2), the RobotEra XHand1 and the
  Sharpa Wave under `assets/bots/hands` (both sides, converted from the vendors' URDF descriptions), and
  FR3 assemblies with the right Wuji Hand 2 (beta 1 and beta 2), Wuji Hand 1, XHand1 and Sharpa Wave under
  `assets/bots/arm_hand_combos`, used by the hand demos.
- `tools/urdf_to_superdex_bot.py`: converts a URDF description into a SuperDex bot package (kinematics
  and inertias through the robotics URDF importer, closed collision meshes, GLB visuals), verified by
  comparing every link transform with the URDF import.
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
- A forward step whose Newton residual is not a finite number is a failure in every mode (a
  NaN residual used to pass the threshold comparison); `TorchRollout` refuses rod actors as force
  actors like the driver and `PolicyRollout` do (a rod used to be given six rigid force DoFs);
  the bridges refuse to build a graph through their backward passes (`create_graph`, Hessians),
  raising instead of returning an incomplete derivative; the closed-loop bridge checks its own
  plain steps and its probe step for non-finite residuals as the driver does.
- Demos: the rigid cubes carry a structured mesh with cells of 12.5 mm, so that contact samples sit
  near their edges and corners (contact acts at surface samples; a 12-triangle cube sank a corner
  10 mm into the ground when tipped before any sample saw it), and the rigid pushes hold the cubes'
  orientation with a running cost, as the policy push did.

### Upstream

- Merged `facebookresearch/project_superdex` `main` through 2026-09-06: wheel projects under their
  components, `fp32`/`fp64` names, contact pair overrides, solver termination classes, sphere-tree
  contact culling, integration bundles.

## [2026-08-24]

- Initial release.
