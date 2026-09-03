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
  and soft material parameters as differentiable tensors.

Supported: rigid, articulated (with pose controllers), soft (tetrahedral FEM) and rod actors;
contact between them and against static colliders (a soft or rod actor collides with rigid and
articulated colliders through its surface samples; mesh colliders and deformable colliders are
rejected in differentiable scenes); node-to-rigid constraints. `make_scene_differentiable`
switches the solver to the settings the adjoint needs (explicit contact normals, exact
gradients of the contact merit, Armijo line search, friction continuation) and disables
recentering of soft actors. Use double precision (`SUPERDEX_PRECISION=double`): the driver
runs on the single-precision build too, but the gradients are then only float32-accurate.

Every gradient path is validated against central finite differences of the same rollout in
`superdex_python/test/diffsim` (run from `superdex_python`):

```bash
SUPERDEX_PRECISION=double python -m unittest discover -s test/diffsim -t .
```

Demos (videos of gradient-based optimization through frictional contact, with a
finite-difference check of the adjoint at the first iteration, `--check`):

```bash
SUPERDEX_PRECISION=double python superdex_physics/examples/example_diffsim_robot_video.py --task all --check
```

The tasks are `reach`, `push`, `push_soft`, `push_multi`, `grasp`, `grasp_soft`, `tendon`
and `haul` (an FR3 arm reaching, pushing a rigid, a soft or two cubes, a 2F-85 gripper
carrying a rigid or a soft cube, a tendon-driven finger with a rod as the cable, and the arm
hauling a box with a cable); `example_diffsim_video.py` holds the actor-level demos.

Known limitations: the adjoint is exact for steps of any size and for consecutive steps of
different sizes, except that pose-controller target gradients keep a residual of about 1e-3
(relative) at steps whose size differs from a neighbour's (pinned by a test);
`get_step_jacobian` assumes two consecutive steps of the same size; the stiffness damping of
soft materials, a deformable actor acting as a collider, and mesh colliders are not
differentiable; a soft body pressed and dragged by a link can trap the forward Newton solve at
isolated steps, which the driver's substepping resolves (see the examples' docstrings).

## License

First-party code in this distribution is Apache-2.0 licensed; see
[LICENSE](https://github.com/facebookresearch/project_superdex/blob/main/LICENSE).
Third-party code and dependencies retain their own terms.
