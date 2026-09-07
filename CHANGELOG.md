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
  `superdex_physics/wheels/superdex-physics/test/diffsim` (145 tests) and the C++ suites.
- Adjoint correctness fixes: rod and soft residual sizing across assemblies, inner-solver
  convergence norm, relative outer tolerance with the true residual reported, stage-start contact
  Jacobians for deformable-vs-dynamic contact, exact adjoints across steps of different sizes
  (previous-delta rescaling, pre-step at the current step size, finite-difference angular
  velocities re-expressed for a changed step size), `get_step_jacobian` for consecutive steps of
  different sizes, and the moving-chart term of external torques on rigid bodies in the adjoint
  operator (the step residual lives in the chart of the iterate, so the step Jacobian is the
  fixed-chart Hessian minus `1/2 [tau]x` on the rotation block; the adjoint solve now uses its
  transpose, by defect correction around the symmetric Krylov solve or directly in the Newton
  outer solver, and `get_step_jacobian` the term itself). Torque gradients went from 2.2e-4
  relative at 0.3 N m (8.6e-4 at 1.2 N m) to below 1e-8 at both on a free cube and are gradchecked
  with every other input group.
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
  simulator, feedback path included (analytic policy gradients).
- Contact-force query adjoint (`diffsim.get_contact_force_world_backward`, the gradient of a
  tactile observation): two missing terms made it exact against a static plane only - 3.7% off
  against a static grid-SDF cube, 15% for a tilting rigid pusher, 26% for the controlled chain
  with Coulomb friction. The derivative of the penalty force `N(d) g(p)` now carries the SDF
  Hessian, `N' g g^T + N H` (the contact detection of a prepared back-propagation computes the
  current-state Hessians; a grid SDF's interpolated gradient turns near edges, where the forward
  solve's quasi-Newton Jacobian leaves the term out), and against a dynamic collider the forces
  turn with it (`d(R_B f)/d delta = delta x R_B f`). All cases now agree with the kinematic
  identity of a free body to 1e-6 (`EngineContactForceAdjointTest`). The adjoint also reaches
  soft bodies: the nodes of a soft-body SDF collider through its mapping and through the
  deformation gradient of its tetrahedra, and the nodes of a soft body whose samples touch the
  queried actor through the samples' interpolation (that direction used to be dropped silently;
  the other refused), and likewise the nodes of a shell or a rod whose samples touch it. A box
  resting on a soft cube agrees with the kinematic identity to 1e-5 in both configurations, a
  cube a rod is dropped on likewise, and a tactile policy on the box passes its
  finite-difference check. Contacts with point-cloud colliders are refused explicitly.
- Release-review fixes (2026-09-07, six findings reproduced against the tree above): (1) a
  contact arriving on a soft-body SDF collider in a differentiable scene aborted the process (the
  stage-start fill of a contact missing at the stage start copied current SDF Hessians a forward
  step never computes); the fill now stores a zero stage-start Hessian, the filled-in normal being
  the current one, and a box dropped onto a soft cube is checked against finite differences
  across the onset. (2) Contact-force losses now reach the contact parameters directly: the
  query `F = sum_s w_s J_s^T f_s(p_s; theta)` depends on the parameters at fixed states, a term
  the state adjoint does not carry (the Coulomb gradient of a sliding cube's force loss was 85%
  too small); the engine differentiates the forces the perturbed residual assemblies store,
  right after the adjoint solve, with the per-contact force adjoints now held apart from the
  forces (`ContactDetectionResult::forceAdjoint`). Checked on the ground, on a two-owner
  cube-cube pair and for a running force loss. (3) Contact-parameter derivatives perturb positive
  coefficients multiplicatively, so a coefficient below the finite-difference step (1e-8) no
  longer samples the negative side of the pair's geometric mean (41% off at mu = 1e-8; now
  1e-5 down to 1e-8). (4) The rollout and policy drivers release every captured state when a
  loss, the adjoint or a native step error raises. (5) `truncation_window` truncates the reverse
  sweep only: the running costs are evaluated live during the forward rollout and the reported
  loss sums every step's. (6) The torch bridges refuse the single-precision engine at
  construction instead of returning float32-accurate gradients as float64 tensors. Found while
  fixing (2): the engine's queries (contact forces, contact points) are outputs of a step, not
  state, so after a state restore they still reported the last forward step - a running
  contact-force loss read the final step's force at every step of the sweep (46% off in the
  gradient). The back-propagation preparation now assembles the prepared state once and
  refreshes the queries, which then match the live values bit for bit.
- Deformable (soft-body SDF) colliders are differentiable: mapped SDF colliders provide SDF
  Hessians, the stage-start query stores the Jacobians of the collider's stage-start mapping
  (`jacWorldFromDofsStageStart`), and both the colliding-side and the collider-side contact
  Jacobians of the previous-state assembly use them. A rigid box resting on a soft cube (the
  box's samples against the cube's mapped SDF) agrees with finite differences to 4e-6 (1e-3 with
  the current mapping in place of the stage-start one), two stacked soft cubes to 2e-7; before,
  such scenes were rejected. Point-cloud (shell, rod) and ROM colliders stay rejected.
- Triangle-mesh colliders (`ColliderType.MESH`, closest-point queries) are differentiable: the
  query computes the signed distance's Hessian by the closest feature (zero on a face,
  `s (I - t t^T - g g^T) / rho` on an edge, `s (I - g g^T) / rho` at a node), which the
  previous-state assembly and the contact-force adjoints need; `make_scene_differentiable` no
  longer refuses them. A rigid cube sliding on a static mesh box agrees with finite differences to
  5e-7, the chain pushing a mesh-collider cube to 1e-4, the contact-force adjoint against a mesh
  wall to 1e-6.
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
