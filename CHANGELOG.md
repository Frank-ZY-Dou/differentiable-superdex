# Changelog – Project SuperDex

All notable changes to this repository will be documented here.

## [1.0.0+diffsim.1] - 2026-09-05

The differentiable-simulation fork of SuperDex 1.0.0 (`+diffsim.N` counts the fork's
releases on that base; these distributions are built from this repository, not from PyPI).


- Differentiable simulation (`superdex.physics.diffsim`, `diffsim_rollout`, `diffsim_torch`):
  the discrete adjoint through the implicit steps now covers rigid, articulated (with pose
  controllers), soft and rod actors, their contacts (soft and rod surface samples against rigid
  and articulated colliders included) and node-to-rigid constraints, with gradients for initial
  states, per-step controller targets and external forces, gravity, contact materials, densities
  and soft material parameters; every path is checked against independent finite differences in
  `superdex_physics/wheels/superdex-physics/test/diffsim` (133 tests) and the C++ suites.
- Adjoint correctness fixes: rod and soft residual sizing across assemblies, inner-solver
  convergence norm, relative outer tolerance with the true residual reported, stage-start contact
  Jacobians for deformable-vs-dynamic contact, exact adjoints across steps of different sizes
  (previous-delta rescaling, pre-step at the current step size, finite-difference angular
  velocities re-expressed for a changed step size), `get_step_jacobian` for consecutive steps of
  different sizes.
- Forward robustness for differentiable scenes: friction continuation
  (`NonLinearSolverParams.friction_continuation_levels`) and failure-adaptive substepping in the
  rollout driver (`DifferentiableRollout(max_substep_levels=...)`, each substep its own adjoint
  step).
- `superdex.physics.utils.penetration.PenetrationChecker`: the interpenetration of a scene from
  the engine's contact samples (deepest sample per actor pair after each step, a report and an
  assertion); the diffsim demos' replays report it and bound it for the rigid tasks.
- `superdex.physics.diffsim_torch.PolicyRollout`: closed-loop rollouts with a torch policy in
  the loop (joint poses, rigid positions and orientations, soft-body centroids and contact
  forces - `ContactForceObservation`, the total contact force on a free rigid body as a tactile
  signal - as observations; pose-controller targets and/or external forces such as joint
  torques as the policy output); the policy parameters receive the loss gradient through the
  simulator, feedback path included (analytic policy gradients). The contact-force observation
  is differentiated through the motion of the observed body (`m dv/dt - m g - f_ext`, exact
  position adjoints; the rollout checks that balance at every step), because the engine's
  contact-force query adjoint, `diffsim.get_contact_force_world_backward`, is exact against
  static colliders only and off by tens of percent against moving ones (pinned by
  `EngineContactForceAdjointTest`; the force Jacobian w.r.t. the contact position is right,
  its mapping onto the moving collider's degrees of freedom is not).
- Examples: `example_diffsim_robot_video.py` (reach, push, soft push, two-cube push, push with a
  feedback policy trained on top of an optimized open-loop plan on three cube starts against an
  open-loop baseline, gripper grasp, soft grasp, five-finger hand grasp, tendon finger, cable haul).
  The rigid tasks use the engine's default contact stiffness (1e9 Pa/m): the 1e6 material of the
  first version let the wrist sink about a centimetre into the pushed cube and the gripper's
  fingertips 15 mm into the grasped one; the soft push keeps a 1e6 contact commensurate with its
  1e5 Pa material (at 1e9 its Newton solve fails), the soft grasp its 1e7 cube. The five-finger
  hand grasp is a fingertip pinch of a 5 cm cube at the default stiffness (its first version only
  held a 7 cm cube by passing the fingers through it at a compliant contact).
  and
  `example_diffsim_sysid.py --mode soft` (Young's modulus and Poisson's ratio identification).
- CI: `diffsim-tests` workflow builds both physics wheels and runs the suite in both precisions.
- Standardized precision names on `fp32` and `fp64`. The public
  `PRECISION_NAME` value now reports the canonical `fp32` or `fp64` name.

## [2026-08-24]

- Initial release.
