# Changelog – Project SuperDex

All notable changes to this repository will be documented here.

## [Unreleased]

- Differentiable simulation (`superdex.physics.diffsim`, `diffsim_rollout`, `diffsim_torch`):
  the discrete adjoint through the implicit steps now covers rigid, articulated (with pose
  controllers), soft and rod actors, their contacts (soft and rod surface samples against rigid
  and articulated colliders included) and node-to-rigid constraints, with gradients for initial
  states, per-step controller targets and external forces, gravity, contact materials, densities
  and soft material parameters; every path is checked against independent finite differences in
  `superdex_python/test/diffsim` (118 tests) and the C++ suites.
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
- `superdex.physics.diffsim_torch.PolicyRollout`: closed-loop rollouts with a torch policy in
  the loop (joint poses, rigid positions and orientations and soft-body centroids as
  observations;
  pose-controller targets and/or external forces such as joint torques as the policy
  output); the policy parameters receive the loss gradient through the simulator, feedback
  path included (analytic policy gradients).
- Examples: `example_diffsim_robot_video.py` (reach, push, soft push, two-cube push, push with a
  feedback policy trained on top of an optimized open-loop plan on three cube starts against an
  open-loop baseline, gripper grasp, soft grasp, five-finger hand grasp, tendon finger, cable haul).
  The rigid tasks use the engine's default contact stiffness (1e9 Pa/m): the 1e6 material of the
  first version let the wrist sink about a centimetre into the pushed cube; the soft push keeps
  a 1e6 contact commensurate with its 1e5 Pa material (at 1e9 its Newton solve fails).
  and
  `example_diffsim_sysid.py --mode soft` (Young's modulus and Poisson's ratio identification).
- CI: `diffsim-tests` workflow builds both physics wheels and runs the suite in both precisions.

## [2026-08-24]

- Initial release.
