# SuperDex Physics

[SuperDex Physics](https://facebookresearch.github.io/project_superdex/physics/) is a
contact-first physics engine purpose-built for tactile manipulation, and applicable wherever
stable contact and accurate sensing matter. This is the simulation backbone of Project
SuperDex.

Provides the `superdex.physics` package: the physics simulation runtime, the viewer, and
the scene/asset utilities.

The published wheel does not carry the debugger GUI client or the mesh CLI helper, so
`superdex.physics.debugger` and `superdex.physics.mesh` are unavailable unless you build
from source. Remote debugging still works: `DebugServer` is part of the core runtime.

```bash
pip install superdex-physics
```

This wheel carries the single-precision native extension. For double precision, install the
`double` extra -- which pulls in `superdex-physics-fp64` -- and select it at import time:

```bash
pip install 'superdex-physics[double]'
export SUPERDEX_PRECISION=double
```

See the [repository README](https://github.com/facebookresearch/project_superdex#readme)
for the full list of SuperDex distributions.

## Differentiable simulation

`superdex.physics.diffsim` exposes the engine's discrete adjoint: the gradient of a loss on
the final (or any) state of a rollout with respect to the inputs of every step, computed by
back-propagating through the implicit (Backward Euler) steps that were actually solved,
contact included. Three layers build on each other:

- `superdex.physics.diffsim` - the per-step engine API (`make_scene_differentiable`,
  `prepare_back_propagate` / `back_propagate`, the `get_*_backward` output adjoints and the
  `set_*_backward` input gradients, `get_step_jacobian`, solver parameters and statistics).
- `superdex.physics.diffsim_rollout.DifferentiableRollout` - a rollout driver: forward with
  per-step state checkpoints, terminal and running losses, the reverse sweep, gradients with
  respect to the initial state, per-step controller targets and external forces, truncated
  sweeps, gradient clipping, solver diagnostics, and failure-adaptive substepping (a step whose
  Newton solve fails is redone as two, four, ... substeps, each its own adjoint step).
- `superdex.physics.diffsim_torch.TorchRollout` - a `torch.autograd.Function` around the
  driver: controls, external forces, gravity, contact materials, densities, soft initial states
  and soft material parameters as differentiable tensors; `PolicyRollout` closes the loop: a
  torch policy maps the observed joint poses, rigid positions and orientations, soft-body
  centroids and contact forces (`ContactForceObservation`, a tactile signal: the total contact
  force on a free rigid body) to the controller targets and/or the external forces (joint
  torques) at every step and its parameters receive the loss gradient through the simulator,
  feedback included (analytic policy gradients).

Supported: rigid, articulated (with pose controllers), soft (tetrahedral FEM) and rod actors;
contact between them and against static colliders (a soft or rod actor collides with rigid and
articulated colliders through its surface samples; mesh colliders and deformable colliders are
rejected in differentiable scenes); node-to-rigid constraints. `make_scene_differentiable`
switches the solver to the settings the adjoint needs (explicit contact normals, exact
gradients of the contact merit, Armijo line search, friction continuation) and disables
recentering of soft actors. Use double precision (`SUPERDEX_PRECISION=double`): the driver
runs on the single-precision build too, but the gradients are then only float32-accurate.

A complete example - a cube pushed to a goal by per-step forces found by Adam through the
simulator (double precision; reaches the goal to within a centimetre and prints the final
distance):

```python
import numpy as np
import torch
import superdex.physics as physics
from superdex.physics import diffsim
from superdex.physics.diffsim_torch import TorchRollout

physics.initialize(num_worker_threads=0)
scene = physics.create_scene("push")
scene.set_gravity([0.0, 0.0, -9.81])
contact = physics.ContactParams(coulomb_friction_coefficient=0.5)
scene.create_rigid_actor(
    name="ground", is_static=True, contact=contact,
    shape=physics.create_plane_shape(normal=[0.0, 0.0, 1.0], distance=0.0),
)
# A dynamic rigid actor needs a surface mesh: a 10 cm cube as a tetrahedral mesh (its
# eight corners, x varying fastest, split into five tetrahedra).
h = 0.05
corners = np.array([[x, y, z] for z in (-h, h) for y in (-h, h) for x in (-h, h)])
tets = np.array([[0, 1, 2, 4], [6, 7, 4, 2], [5, 4, 7, 1], [3, 2, 1, 7], [1, 2, 4, 7]])
cube = scene.create_rigid_actor(
    name="cube", density=1000.0, contact=contact,
    shape=physics.create_tet_mesh_shape(coordinates=corners.ravel(), connectivity=tets.ravel()),
    world_from_local=physics.TransformRT([0.0, 0.0, h]),
)
diffsim.make_scene_differentiable(scene)  # the solver settings the adjoint needs
# The adjoint assumes the step equations hold exactly: converge the forward Newton solve
# far below the engine's default tolerance (1e-3).
params = scene.get_solver_params()
newton = params.non_linear_solver
newton.abs_tol = newton.rel_tol = 1e-10
newton.max_iter = 200
params.non_linear_solver = newton
scene.set_solver_params(params)

goal = np.array([0.3, 0.0, 0.05])

class GoalLoss:  # 0.5 * |final position - goal|^2 and its adjoint
    def value(self) -> float:
        d = np.asarray(cube.get_center_of_mass_transform().translation) - goal
        return 0.5 * float(d @ d)

    def accumulate_output_grad(self) -> None:
        grad = np.zeros(7)  # translation (3) + quaternion (4)
        grad[:3] = np.asarray(cube.get_center_of_mass_transform().translation) - goal
        diffsim.get_center_of_mass_transform_backward(cube, grad)

# Per-step external forces (and torques) on the cube's 6 DoFs as the optimization variable.
rollout = TorchRollout(scene, dt=0.01, num_steps=50, force_actors=[cube], terminal_losses=[GoalLoss()])
forces = torch.zeros(50, 6, dtype=torch.float64, requires_grad=True)
optimizer = torch.optim.Adam([forces], lr=0.2)
for iteration in range(80):  # the cube first has to break static friction, then the force profile is shaped
    optimizer.zero_grad()
    loss = rollout(forces=forces)  # forward rollout + discrete adjoint on backward()
    loss.backward()
    optimizer.step()
print(f"final distance to the goal: {np.sqrt(2 * float(loss)):.4f} m")
rollout.close()
physics.destroy_scene(scene)
physics.shutdown()
```

Every gradient path is validated against central finite differences of the same rollout in
`superdex_physics/wheels/superdex-physics/test/diffsim` (run from that directory):

```bash
SUPERDEX_PRECISION=double python -m unittest discover -s test/diffsim -t .
```

Demos (videos of gradient-based optimization through frictional contact, with a
finite-difference check of the adjoint at the first iteration, `--check`):

```bash
SUPERDEX_PRECISION=double python superdex_physics/examples/example_diffsim_robot_video.py --task all --check
```

The tasks are `reach`, `push`, `push_soft`, `push_multi`, `push_policy`, `grasp`, `grasp_soft`,
`hand`, `tendon` and `haul` (an FR3 arm reaching, pushing a rigid, a soft or two cubes, the push
solved by a feedback policy trained through the simulator on top of an optimized open-loop
plan, on three cube starts at once and against an open-loop baseline of the same architecture,
a 2F-85 gripper carrying a rigid or a soft cube, a five-finger DG-5F hand carrying a cube, a tendon-driven finger with a rod as the
cable, and the arm hauling a box with a cable); `example_diffsim_video.py` holds the actor-level demos, and
`example_diffsim_sysid.py` identifies parameters from observed trajectories (a sliding cube's
friction coefficient and density, or with `--mode soft` a dropped soft cube's Young's modulus and
Poisson's ratio, recovered to 0.01%).

The rigid tasks use the engine's default contact stiffness (1e9 Pa/m, `physics.ContactParams()`):
a softer material lets a position-controlled arm sink visibly into what it pushes. The soft
tasks use a stiffness commensurate with their material (1e6 on a 1e5 Pa cube), the level at
which their Newton solve still converges to the tolerance the adjoint needs. The five-finger
hand grasp is a fingertip pinch (fingers 2-4 on the far face of a 5 cm cube, the thumb on the
near face) at the default stiffness: the DG-5F's thumb cannot oppose the finger pads along the
finger direction by more than about 5 cm, so a palm-down power wrap of a 7 cm cube is not
reachable, and the first version of the demo only held its cube by passing the fingers through
it at a compliant contact.

`superdex.physics.utils.penetration.PenetrationChecker` measures the interpenetration of a
scene from the engine's contact samples (the deepest sample per actor pair after each step,
a report, an assertion) - a penalty contact overlaps under load, about 1-3.5 mm at the default
stiffness in these demos, and the checker is how the demos' replays keep it bounded (5 mm for
the rigid tasks). What the samples do not see, it does not see: the FR3 wrist links' render
models extend up to a centimetre beyond their collision hulls in places, so a wrist that
visibly enters a cube may be overlapping less than it looks.

What the demos reach (40 iterations unless noted; the loss is half the squared distance of
the manipulated object to its goal, so 1e-4 is about 1.4 cm and 1e-6 about 1.4 mm; every run
starts with the adjoint checked against central finite differences along its own gradient):

| task | what is optimized | loss at start | loss at the end | adjoint vs FD |
|---|---|---|---|---|
| `reach` | joint targets, free motion | 6.7e-2 | 2.0e-4 | 1.2e-5 |
| `push` | joint targets through frictional contact | 7.4e-3 | 2.2e-5 | 6.0e-6 |
| `push_multi` | joint targets through two contacts (cube pushes cube) | 6.8e-3 | 3.2e-5 | 4.4e-6 |
| `push_soft` | joint targets, FEM cube, substepped | 7.6e-3 | 1.3e-4 | 6.8e-5 |
| `push_policy` | MLP weights, closed loop, three cube starts (100 it.) | 3.8e-3 (mean) | 8.5e-5; 0.1 / 0.6 / 0.8 cm to the goal | 2.0e-3 (mean over starts) |
| `grasp` | carry knots, 2F-85 gripper, closed-loop linkage | 1.1e-2 | 2e-6 | 1.4e-4 |
| `grasp_soft` | carry knots, FEM cube in the gripper | 1.1e-2 | < 1e-6 | 1.3e-3 |
| `hand` | carry knots, DG-5F fingertip pinch, 27 DoFs | 1.1e-2 | < 1e-6 | 3.0e-5 |
| `tendon` | tendon pull of a rod-driven finger | 9.9e-4 | < 1e-6 | 1.1e-4 |
| `haul` | joint targets through a cable to a box | 6.6e-3 | 2.5e-3 | 2.3e-5 |

Cost (one thread, double precision, a 2026 desktop CPU): the FR3 arm pushing a cube
(75 steps of 20 ms) runs at about 6 ms per forward step and 4 ms per adjoint step, a
forward-plus-backward rollout in under a second; with a soft cube (64 nodes) the adjoint step
costs about 2.5 forward steps. The demo optimizations above (40 iterations, with the replays
that record the videos) take between half a minute (cable haul) and three minutes (soft grasp).

Known limitations: the adjoint is exact for steps of any size and for consecutive steps of
different sizes (at a step-size change the engine re-expresses the previous finite-difference
angular velocities for the new size, keeping the angular rate; `get_step_jacobian` handles it
the same way, and the adjoint operator carries the moving-chart term of external torques on
rigid bodies, so torque gradients are exact to 1.6e-7 at 0.3 N m and 3e-5 at 1.2 N m on a 0.2 m
cube; the contact-force query adjoint is exact against static and moving colliders alike,
checked against the kinematic identity of a free body); the
stiffness damping of soft materials, a deformable actor acting as a collider (its stage-start
contact Jacobians are not differentiated: 1e-3 relative error against finite differences for a
rigid box on a soft cube, so such scenes are rejected), and mesh colliders (no SDF Hessians) are
not differentiable; a soft body pressed and dragged by a link can trap the forward Newton solve at
isolated steps, which the driver's substepping resolves (see the examples' docstrings). Losses
through frictional contact are piecewise smooth: at steps where the forward Newton solve is
nearly degenerate (100+ iterations to a 1e-9 residual, an arm pushing a cube over the ground)
a parameter perturbation of a few 1e-8 can land it on another local solution, a jump of about
1e-7 in the loss (5e-6 m in a cube position). The adjoint is the exact derivative of the branch
taken (finite differences agree to 1e-3 at eps 1e-8, and to 1e-5 where the loss is smooth); a
finite-difference check of such a loss needs eps at or below 1e-7 and a self-consistency test
across epsilons, as the examples' `--check` reports. For optimization the jumps are noise
(an Armijo line search stalls on them; fixed or decaying steps step over them), but a step
that loses the contact lands on the zero-gradient plateau of the untouched cube, from which
no gradient recovers: the feedback-policy demo halves any step that raises the loss by more
than half, the open-loop demos clip the gradient.

## License

First-party code in this distribution is Apache-2.0 licensed; see
[LICENSE](https://github.com/facebookresearch/project_superdex/blob/main/LICENSE).
Third-party code and dependencies retain their own terms.
