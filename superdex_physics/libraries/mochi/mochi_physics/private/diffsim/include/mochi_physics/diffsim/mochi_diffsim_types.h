/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <mochi_physics/mochi_physics_experimental.h>

#include <mochi_core/solvers/nonlinear_solver_params.h>
#include <mochi_core/utils/verbosity_params.h>

namespace mochi::diffsim {

struct BackPropagationSceneStats {
  double totalDurationSec = 0.0;
  double solveDurationSec = 0.0;
  int maxOuterIters = 0;
  // Final residual norm of the adjoint solve. With BackPropagationSolverParams::validateFiniteDiff
  // set, this is the true residual |H z - rhs| of the returned solution, recomputed with a fresh
  // Hessian-vector product (MINRES's implicit residual can under-report); otherwise the solver's
  // own estimate.
  double residualNorm = 0.0;
  bool finiteDiffValid = true;
  // Relative asymmetry of the adjoint operator H (the step Jacobian), measured after the solve as
  // |rhs.(H z) - z.(H rhs)| / mean(|rhs.(H z)|, |z.(H rhs)|) with two extra Hessian-vector
  // products; maximum across islands. Only computed with validateFiniteDiff set (0 otherwise).
  // With the finite-difference operator the probe has a noise floor set by the products' own
  // error (measured 1.3e-4 at the default epsilon 1e-8 on an articulated-vs-rigid frictional
  // island, 1.3e-3 at 1e-7, i.e. scaling with the epsilon; the gradients themselves are
  // unaffected at that level); with the
  // analytic operator it is exact (1e-15). Values well above the floor mean the residual is not
  // the gradient of one merit function, and the symmetric adjoint solve (PCG / MINRES) is then
  // only approximate.
  double hessianAsymmetry = 0.0;
};

struct BackPropagationSolverParams {
  VerbosityLevel verbosity = NonLinearSolverParams{}.verbosity;
  bool useNewtonOuterSolver = false;
  int outerSolverMaxIter = 10;
  real outerSolverAbsTol = 1e-3_r;
  real outerSolverRelTol = 1e-10_r;
  NonLinearSolverConvergenceMode outerSolverConvergenceMode =
      NonLinearSolverConvergenceMode::Global;
  real innerSolverAbsTol = 1e-10_r;
  real epsFiniteDiff = kDefaultBackPropagationEpsFiniteDiff;
  bool validateFiniteDiff = false;
  // Experimental: use the analytically assembled Hessian (psdDRes = false, exact
  // saturation Hessians) as the outer-solve operator instead of finite-difference
  // Hessian-vector products (Krylov outer solver only). The assembly is exactly
  // symmetric but Gauss-Newton-grade everywhere (test/diffsim/test_analytic_hvp.py,
  // measured with two-step-size rollout finite differences, 2026-09-01): on a rigid
  // cube sliding on a static plane its gradients are off by 7e-4 (rich friction) to
  // 4e-2 (frictionless) relative, where the finite-difference operator is exact to
  // 1e-7; articulated islands and dynamic-dynamic contact coupling are worse (5e-2
  // and up). Use it as a fast approximate operator only; the finite-difference
  // operator is the accurate one. With validateFiniteDiff also set, each solve
  // cross-checks the analytic operator against one finite-difference product and
  // reports mismatches through BackPropagationSceneStats::finiteDiffValid.
  bool useAnalyticHvp = false;
};

} // namespace mochi::diffsim
