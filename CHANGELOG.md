# Changelog – Project SuperDex

All notable changes to this repository will be documented here.

## [Unreleased]

### Fixed

- The mass-damping parameter gradient of a soft actor near zero. The material parameters are
  differentiated by finite differences of the assembled residual with a step of about 1e-3, and
  the damping term is gated at zero (a non-positive coefficient disables it); the lower sample of
  a central difference at any positive coefficient below the step crossed that gate and the
  gradient came out at about half its value (49.95 percent low at 1e-6, 45 percent at 1e-4 on
  the extracted stencil; only the coefficient exactly zero had been treated). The lower sample
  now stays in the domain (a right-sided difference, exact for a term affine in the coefficient),
  and the same rule keeps Young's modulus and the density positive.
  `test_diffsim_torch.py::SoftMassDampingStencilTest` compares the gradient at 0, 1e-8, 1e-6,
  1e-4 and 5e-4 with the one at 2e-3, where an in-domain central difference of the rollout
  loss confirms it.
- The adjoint operator's symmetry probe (`validate_finite_diff`) took rhs.(J^T z) on one side
  and z.(H rhs) on the other, so with an external torque it reported the antisymmetric
  moving-chart term 1/2 [tau]x as an asymmetry of the step Jacobian H (18 percent on the
  audit's synthetic case). It now probes H alone;
  `test_diffsim_rollout.py::AdjointSymmetryProbeTest` (a rigid cube and a Free root under a
  torque) reads it below 5e-3.
- `PolicyRollout`: a policy whose control at a step is a constant tensor (no graph) made the
  step's vector-Jacobian product raise inside torch; the step now contributes zero, as it
  should, and a policy may switch between constant and differentiable controls
  (`PolicyRolloutConstantActionTest`, with a finite-difference check of the mixed case).
- The rotation gradients of Free and Spherical joints. The chart transport of an output gradient
  (`diffsim.get_articulated_pose_backward`, the seed of every rollout loss on a joint pose)
  applied the rotation-vector Jacobian where a gradient needs its transpose, so a round trip
  through the two chart conversions returned the gradient rotated by the transpose of the joint
  rotation instead of itself, and every gradient reaching such a joint's rotation was off by the
  joint rotation: tens of percent at 0.3 rad, more than the gradient at 1.5 rad, and a smaller
  error growing with the rotation a step produces even from the identity (the transport is
  evaluated at the final pose). The velocity adjoints
  (`set_articulated_joint_velocities_backward`, `set_articulated_target_velocity_backward`)
  also applied the pose chart transport to what is a Lie-algebra quantity (the joint's angular
  velocity in its outer frame, which the Jacobian maps to the link twists and the step integrates
  as a left rotation increment). The forward was consistent throughout; revolute and prismatic
  joints, which have no rotation chart, were exact. The engine's own C++ conversion test had
  not caught the transposition because its merit was the pose's squared norm, whose gradient is
  parallel to the rotation vector, an eigenvector of both chart Jacobians; it now adds a random
  linear term. Three smaller defects of the same joints surfaced underneath and are fixed with
  it: the articulated velocity adjoint chained the links' and joints' rotation increments as
  dt (the rigid one chains through the increment a set angular velocity stands for,
  DR = exp(phi), phi = asin(dt |omega|) omega/|omega|; that chain is now shared, applied per
  link before the Jacobian transposes it and per Free/Spherical joint, and the initial-pose
  adjoint's Jacobian term uses the same chained adjoint), leaving an error of order dt |omega|
  orthogonal to omega (5e-4 at dt = 0.01, |omega| = 0.3); the controller paths chained the
  previous target's dependence on a velocity (target_old = exp(-dt v) target) and on the new
  target as the identity, missing the increment's left Jacobian and the rotation R(-dt omega)
  on the rotations (1e-3 and 5e-3); and the adjoint operator carried the moving-chart term of an
  external torque, J^T = H + 1/2 [tau]x, for standalone rigid actors only, not for the torque
  on a Free or Spherical joint, which enters through the same merit on the joint transform
  (4e-5 at dt = 0.01, 0.3 N m). Initial pose, initial velocity, external forces and torques on
  every dof, controller targets and target velocities of a Free root in free fall and of a
  Spherical joint on a rotated parent (with a rotated rest frame, a pose controller, joint
  inertia and friction) now agree with central finite differences to 1e-6 relative, at
  rotations of 1.2 rad and near the identity.
- The URDF importer expresses a link's inertia tensor in the link frame: the tensor URDF gives
  in the inertial frame is rotated by the `rpy` of `<inertial><origin>`, which the importer used
  to parse and drop. `tools/urdf_to_superdex_bot.py` no longer reports such links.
- The initial-pose input adjoint of articulated actors
  (`diffsim.set_articulated_pose_from_joints_backward`) missed the dependence of the links'
  previous deltas on the pose: the link velocities are J(q) v, so a loss also depends on the
  pose through the Jacobian. The term dt (d(J(q) v)/dq)^T g is now added (central differences
  of the Jacobian in the dofs), and in a differentiable scene the pose setters
  (`set_articulated_pose_from_joints`, `set_articulated_pose_from_links`,
  `set_root_transform`) re-derive the link velocities from the joint velocities, as the
  velocity setter does, so the state and the adjoint agree whatever the order of the setters
  (other scenes keep the engine's behavior). On a revolute-prismatic chain the gradient was
  off by a tenth, on a revolute pendulum by 1e-4; it agrees with finite differences to 1e-6
  now on revolute and prismatic chains (the rotation charts of Free and Spherical joints are
  fixed above). Initial velocities of zero were unaffected. The stored
  double-precision reference gradients were regenerated.
- `tools/urdf_to_superdex_bot.py`: a link without a collision mesh gets no shape, and the engine
  gives a shapeless link no mass; the tool now folds the inertial of such a link into its parent
  across a fixed joint (parallel-axis update in the parent's frame) and reports a shapeless link
  on a moving joint, which stays massless. A URDF whose root is a bare `world` frame (no
  inertial, no mesh) now yields a fixed base (a Hard world joint) instead of a floating one;
  `--base fixed|floating` overrides the detection. The joint dynamics the importer reads
  from the URDF (viscous damping and Coulomb friction, joint inertia, limit stiffness and
  damping) are written to the package. The package verification moves every joint inside its
  limits instead of assuming a floating root, so fixed-base robots verify too. Mimic joints,
  which the importer does not honor, are reported.

### Added

- `diffsim.get_contact_points_backward(actor, grad_output)`: the per-contact backward of
  `get_contact_points_world`, one 3-vector per reported point (in the query's order) with respect
  to the point's reported force - the quadrature-weighted world force on `actor_a`, so the points
  where the actor is `actor_b` carry the force on the other body's sample. Each point is matched
  to the contact containers by its actor pair and sample index, and its adjoint w_s J_B lambda is
  accumulated as the total-force backward accumulates one gradient for all points; seeding every
  point with the same gradient reproduces `get_contact_force_world_backward` (2026-09-09,
  `EngineContactPointsAdjointTest`: 8e-10 relative between the two seedings on the chain pushing
  the cube, and a loss weighting every point by a fixed function of its sample index and role
  agrees with central finite differences of the rollout to 1e-5 along the gradient). This is the
  engine side of a tactile array: a loss on the per-taxel readout of a fingertip (a fixed
  distribution of each contact over the taxels it faces) reaches the controls through the
  per-contact forces and, through `get_root_transform_backward`, the link rotation that turns
  them into the sensor frame. The positions of the points carry no adjoint (a sample is a fixed
  point of its rigid part).
- The differentiation contract of the rollouts, reported and enforced instead of assumed. The
  adjoint is the derivative of the step equations at a solution, so the driver and both torch
  bridges now (1) accept `forward_residual_tolerance`, which makes a step whose Newton solve ended
  unconverged above it raise `ForwardSolveError` in a rollout without substepping (with
  substepping the substep tolerance already is the contract); (2) check every adjoint solve before
  aggregating its statistics: a non-finite true residual raises `AdjointSolveError`, and so does,
  by default (`require_adjoint_convergence`), a solve the engine reports as not converged; (3)
  reject a non-finite gradient before returning it; (4) report each condition in its own field of
  `RolloutResult` and `PolicyRolloutResult` (`forward_converged`, None when no tolerance was
  given; `max_forward_residual`; `adjoint_converged`; `adjoint_residual_threshold`;
  `adjoint_residual_floor`; `gradients_finite`; `fd_validation_ran`, and `fd_validation_passed`,
  None when the engine's finite-difference self-check did not run, so that `fd_valid`'s default of
  True is not read as a pass). A NaN residual used to disappear into `max(...)`. The engine
  reports the verdict itself: `BackPropagationSceneStats.converged`, `residual_threshold` and
  `residual_floor`. Every adjoint solve is now checked against its true residual, evaluated with
  fresh products of the full operator J^T (the Krylov solve stops on a recurrence residual that
  the finite-difference noise lets drift from the true one). On an island with an external torque
  the defect correction of the chart term runs to convergence, while each round still lowers the
  true residual (the former fixed budget of eight rounds is now a stall test and a budget of 32);
  on an island without one the plain solve stands, and rounds are run only to bring a solve that
  is above the acceptance threshold within the contract (a round on a solve at the noise floor of
  the products lowers the measured residual without bringing the solution closer, and moved
  gradients on stiff islands by 1e-5 relative). A solve is converged when its finite true residual
  is at or below the larger of the outer threshold max(abs, rel |rhs|) and 1024 times (64 in
  single precision) the operator's round-off level, machine epsilon over the finite-difference
  step times |rhs| + |J^T z| (2e-8 relative in double precision at the default step, 6e-4 in
  single: the round-off of a central difference of the residual, which depends on the precision
  and the step and not on the loss, so no fixed tolerance expresses it and a request below it used
  to be met only by the recurrence residual; the factor covers what a solve reaches above the
  level through the state's magnitude, the island's stiffness and its conditioning, up to 126
  levels in double precision on velocity gradients through rigid contact and 284 on a rod, 33 in
  single precision, while a failed solve sits orders of magnitude above; measuring the level from
  two quotients at different steps was tried and dropped, the rounding of the two cancels for a
  residual that is linear along the solution). A solve that stopped on its budget, diverged, or
  stalled above the threshold is reported as not converged rather than returned as a gradient. The
  scene statistics report the residual, the threshold and the level as one consistent triple: over
  all islands when the step converged (so that `residual_norm` is at or below
  `residual_threshold`), and those of the island that failed worst when it did not. The
  double-precision reference gradients of `test_precision_reference.py` are unchanged to the last
  bit: the healthy solves of the suite are the ones the engine produced before (a version of the
  refinement that ran rounds on torque-free islands moved the free chain on the plane by 3.4e-6
  relative, where the stored and the moved gradient were 1.6e-6 and 2.5e-6 from a central
  difference stable to 1e-9 across four step sizes, the accuracy the operator's 1e-8 noise leaves
  at that island's conditioning).
  `test_diffsim_rollout.py::DifferentiationContractTest` and
  `test_diffsim_torch.py::PolicyRolloutContractTest` starve the forward and the adjoint
  solvers, inject NaN into the statistics and into a gradient, request a tolerance below the
  floor, and read the fields back.
- `test/diffsim/test_diffsim_rollout.py::ArticulatedRotationChartTest`: the driver's initial-pose,
  initial-velocity, external-force and controller-target gradients against finite differences
  for a Free root in free fall (`scenes.free_chain`, at 1.2 rad and near the identity) and for a
  Spherical joint on a rotated parent with a rotated rest frame
  (`scenes.chain_revolute_spherical`, with and without a pose controller, and with joint inertia
  and viscous friction); `test_diffsim_gradients.py::TargetVelocityBackwardTest` gains the
  Spherical joint.
- The rollout driver reads external-force gradients on every joint dof of an articulated actor
  (`ActorGradients.external_forces`, `force_dofs`), the torques of Free and Spherical joints
  included; it used to read the single-dof joints only. `TorchRollout` and `PolicyRollout`
  force actors follow.
- `superdex_robotics/test_python/test_superdex_robotics.py::UrdfInertialImportTest`: a rotated
  inertial origin lands in the link frame as R I R^T; an unrotated one is unchanged.
- `diffsim_torch.PolicyRollout`: a policy may return `(controls, aux)` and `aux_losses(step, aux)`
  adds a per-step loss on the side output; its gradient is folded into the per-step
  vector-Jacobian product, so it reaches the parameters and the observation feedback path.
  `PolicyRolloutResult.aux_loss` reports the auxiliary share of the loss.
  `test/diffsim/test_diffsim_torch.py::PolicyRolloutTest::test_aux_loss_policy_vs_fd` checks
  both heads' parameter gradients against finite differences of the whole objective.
- `test/diffsim/test_diffsim_rollout.py::ArticulatedInitialPoseTest`: the driver's initial-pose
  gradient of a torque-driven articulated actor against finite differences, on the revolute
  pendulum with and without its pose controller and on the new
  `scenes.chain_revolute_prismatic` (a prismatic joint after a revolute one), with the initial
  state set as pose then velocities, velocities then pose, and the pose alone on a state that
  has stepped.

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
  rollout driver, whose `observe_substep(step, sub_dt)` and `observe_initial()` hooks
  (forwarded by the torch bridges) run a monitor on the final state of every accepted
  (sub)step of every rollout and on its initial state.
- `superdex.physics.utils.penetration.PenetrationChecker`: the interpenetration of a scene
  measured from the engine's contact samples, with a report and an assertion. It watches every
  non-static rigid, soft, shell or rod actor by default, with or without a collider of its own
  (a soft actor created from Python has none and still emits samples against the ground), tells
  actors apart by handle, counts a sample listed by both bodies' queries once, and gives no safe
  verdict for an unobserved scene, a non-finite sample or a non-finite limit. A sampled measure,
  not a collision certificate.
- Examples in `superdex_physics/examples`: `example_diffsim_throw.py` (the per-step API by hand),
  `example_diffsim_video.py` (a rigid throw and finite-element soft-body tasks: a jelly landing on
  a target and a jelly shoving a jelly), `example_diffsim_tactile.py` (differentiable tactile control
  of the XHand1's 600-taxel fingertip map), `example_diffsim_sysid.py`
  (friction and density, or soft material parameters, identified from trajectories),
  `example_diffsim_ik.py` (inverse kinematics through the simulator: joint targets solved by
  L-BFGS on the settled pose, compensating the controller's sag under gravity, and with a
  contact-force objective against a box) and `example_diffsim_robot_video.py` (fifteen manipulation
  tasks on video, with finite-difference checks, among them grasps with six hands). With `--export-scenes` the demos also write the recorded frames' bodies, meshes,
  textures and camera, and `render_diffsim_blender.py` renders them with Blender.
- Assets: hand packages for the Wuji Hand 1, the Wuji Hand 2 (beta 2), the RobotEra XHand1 and the
  Sharpa Wave under `assets/bots/hands` (both sides, converted from the vendors' URDF descriptions),
  the XHand1 v1.3 package with fingertip tactile taxel layouts (`assets/bots/hands/xhand1_official`), and
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
