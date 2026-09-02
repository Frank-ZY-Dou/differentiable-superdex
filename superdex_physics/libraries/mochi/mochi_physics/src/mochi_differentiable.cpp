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

#include "mochi_differentiable.h"

#include "mochi_actor_convergence.h"
#include "mochi_common_components.h"
#include "mochi_articulated_body.h"
#include "mochi_constraint.h"
#include "mochi_materials.h"
#include "mochi_rigid.h"
#include "mochi_simulation.h"
#include "mochi_soft.h"
#include "mochi_solve.h"
#include "mochi_step.h"

#include <mochi_physics/diffsim/mochi_diffsim_types.h>

#include <mochi_core/solvers/linear_solver.h>
#include <mochi_core/solvers/newton_solver.h>
#include <mochi_core/utils/assembly_params.h>
#include <mochi_core/utils/task_scheduler.h>

#include <cmath>
#include <limits>
#include <unordered_set>

using namespace mochi;

namespace {
// Traits to map GradTarget to container, offset and size components
template <GradTarget kGradTarget>
struct GradTargetTraits {};

#define MOCHI_SPECIALIZE_GRAD_TARGET_TRAITS(kGradTarget, kContainer, kSize, kOffset) \
  template <>                                                                        \
  struct GradTargetTraits<GradTarget::kGradTarget> {                                 \
    using ContainerT = kContainer;                                                   \
    using SizeT = kSize;                                                             \
    using OffsetT = kOffset;                                                         \
  }
// clang-format off
MOCHI_SPECIALIZE_GRAD_TARGET_TRAITS(Current, CDiffContainerState, CActorDofInfo, CDofOffset);
MOCHI_SPECIALIZE_GRAD_TARGET_TRAITS(Previous, CDiffContainerState, CActorDofInfo, CDofOffset);
MOCHI_SPECIALIZE_GRAD_TARGET_TRAITS(PreviousDelta, CDiffContainerDerivedState, CActorDerivedStateInfo, CDerivedStateOffset);
// clang-format on
#undef MOCHI_SPECIALIZE_GRAD_TARGET_TRAITS
} // namespace

template <GradTarget kGradTarget>
static void GetContainer(
    ColumnVectorView<real> outData,
    typename GradTargetTraits<kGradTarget>::SizeT const& size,
    typename GradTargetTraits<kGradTarget>::OffsetT const& offset,
    typename GradTargetTraits<kGradTarget>::ContainerT const& container) {
  outData.MiddleRows(offset.dofsOffset, size.dofsSize) = container;
}

template <GradTarget kGradTarget>
static void SetContainer(
    ColumnVectorView<real const> data,
    typename GradTargetTraits<kGradTarget>::SizeT const& size,
    typename GradTargetTraits<kGradTarget>::OffsetT const& offset,
    typename GradTargetTraits<kGradTarget>::ContainerT& outContainer) {
  outContainer = data.MiddleRows(offset.dofsOffset, size.dofsSize);
}

// Update all target pose gradients: accumulate current, set previous, compute propagated.
static void UpdateTargetPoseGrad(
    ColumnVectorView<real const> currentGrad,
    ColumnVectorView<real const> previousGrad,
    uint64_t stepCounter,
    CActorDiffInputInfo const& size,
    CDiffInputOffset const& offset,
    CTargetOwners const& owner,
    CDiffTargetPoseGrad& outTargetPoseGrad) {
  MOCHI_ASSERT_VERBOSE(owner.oldPoseStep < stepCounter, "Inconsistent step counter");
  MOCHI_ASSERT_VERBOSE(owner.velStep < stepCounter, "Inconsistent step counter");
  MOCHI_ASSERT_VERBOSE(owner.newPoseStep < stepCounter, "Inconsistent step counter");
  if (owner.velStep > owner.oldPoseStep) {
    MOCHI_LOG_WARNING_ONCE(
        "The target velocity of the pose controller was set without setting the target pose. Differentiability gradients may be incorrect.");
  }

  // Accumulate current input gradient (adds propagated from previous step).
  outTargetPoseGrad.current =
      outTargetPoseGrad.propagated + currentGrad.MiddleRows(offset.dofsOffset, size.dofsSize);

  // Set previous input gradient.
  outTargetPoseGrad.previous = previousGrad.MiddleRows(offset.dofsOffset, size.dofsSize);

  // Compute what should propagate to the previous step.
  // Propagate oldPose gradient only if oldPose was inherited (not set at this step).
  if (owner.oldPoseStep < stepCounter - 1) {
    outTargetPoseGrad.propagated = outTargetPoseGrad.previous;
  } else {
    outTargetPoseGrad.propagated.SetZero();
  }
  // Propagate newPose gradient only if newPose was inherited (not set at this step).
  if (owner.newPoseStep < stepCounter - 1) {
    outTargetPoseGrad.propagated += outTargetPoseGrad.current;
  }
}

// Implement d2merit/dtargetdq * vector as a finite difference approximation (0.5 / eps) *
// (dmerit/dtarget(q0 + eps * vector) - dmerit/dtarget(q0 - eps * vector)).
// For kGradTarget = GradTarget::Current, dmerit/dtarget must be transported to q0.
static void GetHessianVectorProduct(
    entt::registry& reg,
    entt::entity island,
    GradTarget gradTarget,
    SnleProblem<real>& problemForward,
    ColumnVectorView<real const> vector,
    ColumnVectorView<real> outHvp) {
  if (IsZero(vector)) {
    outHvp.SetZero();
    return;
  }

  // Stack memory for local vector data (6 x 256 elements)
  MOCHI_FILO_STACK_ALLOCATOR(allocator, 6 * 256 * sizeof(real));

  // Set up the gradient assembly function
  AssemblyParams params = {
      .assemObj = false, .assemRes = true, .assemDRes = false, .gradTarget = gradTarget};
  problemForward.SetAssemblyFunction(
      [&](SnleProblem<real>& problem, AssemblyParams const& /* params */) {
        solver::AssembleIslandPipeline(reg, island, params, problem);
      });

  // Size eps such that each pose component changes by epsFiniteDiff.
  auto const& solverParams = reg.ctx<CBackPropagationSolverParams const>();
  int const numDofs = problemForward.GetDofsSize();
  auto const eps = Sqrt(static_cast<real>(numDofs)) * solverParams.epsFiniteDiff /
      (vector.Norm() + std::numeric_limits<real>::min());
  ColumnVector<real> delta(numDofs, &allocator);

  // Lambda for gradient evaluation
  int const solutionSize = problemForward.GetSolutionSize();
  ColumnVector<real> refSolution(solutionSize, &allocator);
  refSolution = problemForward.solution;
  auto evalGradient = [&](ColumnVectorView<real> outGradientLocal) {
    solver::PostNewIncrementLocalPipeline(reg, island, delta, problemForward.solution);
    problemForward.InvalidateCachedData();
    problemForward.UpdateResidual();
    outGradientLocal = problemForward.GetResidual();
    if (gradTarget == GradTarget::Current) {
      // Gradients computed with Lie derivatives use local parameterizations of 3D rotations. When
      // the target is GradTarget::Current, the gradient is evaluated at slightly offset rotations,
      // hence the local parameterizations are slightly incorrect. The gradient must be transported
      // to account for the correct local parameterization of rotations.
      auto const& descendants = reg.get<CIslandDescendants const>(island);
      ecs::InvokeForEach(
          rigid::TransportGradient,
          reg,
          descendants.rigidActors,
          AsConstView(delta),
          outGradientLocal);
      ecs::InvokeForEach(
          articulated::compound::TransportGradient,
          reg,
          descendants.compoundActors,
          AsConstView(delta),
          outGradientLocal);
    }
    problemForward.solution = refSolution;
  };

  // Approximate Hessian-vector product with central finite differences of gradient
  ColumnVector<real> auxGrad(outHvp.Rows(), &allocator);
  auto evalHvpAtEps = [&](real epsScale, ColumnVectorView<real> outGrad) {
    delta = (epsScale * eps) * vector;
    evalGradient(outGrad);
    delta *= -1_r;
    evalGradient(auxGrad);
    outGrad -= auxGrad;
    outGrad *= (0.5_r / (epsScale * eps));
  };
  evalHvpAtEps(1_r, outHvp);

  // Validate the difference quotient and refine it where it has not converged (only with
  // validateFiniteDiff set). For a smooth residual the truncation error is O(eps^2), so the
  // quotients at eps and eps/2 agreeing to kFiniteDiffTol bounds the error of the finer one by
  // about a third of the tolerance. Where they disagree (stiff contact terms and contact
  // transitions: on the FR3 + 2F-85 grasp island a handful of steps per backward at the default
  // epsilon, all of them clean one decade finer), halve the step until two consecutive
  // quotients agree, at most kMaxRefinements times, and return the finer product of the
  // agreeing pair. A product that never converges is recorded in the per-island stats
  // (aggregated into BackPropagationSceneStats::finiteDiffValid via logical AND) and the finest
  // product is returned.
  if (solverParams.validateFiniteDiff) {
    real constexpr kFiniteDiffTol = 1e-2_r;
    int constexpr kMaxRefinements = 4;
    auto& islandBackPropStats = reg.get<CIslandBackPropSolverStats>(island);
    ColumnVector<real> finer(outHvp.Rows(), &allocator);
    ColumnVector<real> diff(outHvp.Rows(), &allocator);
    real epsScale = 0.5_r;
    evalHvpAtEps(epsScale, finer);
    bool converged = false;
    int refinements = 0;
    while (true) {
      diff = finer;
      diff -= outHvp;
      real const err = diff.Norm() / (finer.Norm() + std::numeric_limits<real>::min());
      if (err <= kFiniteDiffTol) {
        converged = true;
        break;
      }
      if (refinements == kMaxRefinements) {
        break;
      }
      ++refinements;
      outHvp = finer;
      epsScale *= 0.5_r;
      evalHvpAtEps(epsScale, finer);
    }
    outHvp = finer;
    if (!converged) {
      islandBackPropStats.finiteDiffValid = false;
      if (solverParams.verbosity >= VerbosityLevel::Verbose) {
        MOCHI_LOG(
            "Finite difference did not converge after %d halvings of eps (tol=%f)",
            refinements,
            kFiniteDiffTol);
      }
    } else if (refinements > 0 && solverParams.verbosity >= VerbosityLevel::Verbose) {
      MOCHI_LOG("Finite difference refined: %d halvings of eps", refinements);
    }
  }
}

static void WriteToActorResidual(
    ColumnVectorView<real const> islandResidual,
    SnleProblem<real>& outProblem,
    ecs::Excluded<TagArticulatedLinkActor>,
    CActorConvergenceWeights const& weights,
    CActorDofInfo const& dofInfo,
    CDofOffset const& dofOffset,
    CActorSnle& outSnle) {
  auto& outResidual = outSnle.UseReduced() ? outSnle.reducedResidual : outSnle.fullResidual;
  outResidual = islandResidual.MiddleRows(dofOffset.dofsOffset, dofInfo.dofsSize);
  outProblem.actorResiduals.emplace_back(dofOffset.dofsOffset, &outResidual);
  outProblem.actorConvergenceWeights.emplace_back(dofOffset.dofsOffset, &weights.values);
}

// Assemble the exact (non-PSD-projected) Hessian d2merit/dq2 of the island at the
// prepared states, as an owned dense copy. This is the same assembly the forward-mode
// step-Jacobian path relies on (psdDRes = false). The problem's cached dresidual is
// left holding this exact assembly; re-assemble before using it for anything else.
static Matrix<real> AssembleExactHessian(
    entt::registry& reg,
    entt::entity island,
    SnleProblem<real>& problemForward) {
  AssemblyParams params = {
      .assemObj = false,
      .assemRes = false,
      .assemDRes = true,
      .psdDRes = false,
      // Exact saturation Hessians as well: the fitted variants are Newton
      // stabilization devices, not derivatives of the actual residual.
      .fittedSaturationHessian = SaturationHessianParams::All(false)};
  solver::AssembleIslandPipeline(reg, island, params, problemForward);
  return ToMatrix(problemForward.GetDResidual());
}

// out = matrix * in, written out explicitly against the column-major storage so the
// Krylov operator has no dependency on expression-template overloads.
static void ApplyDense(
    MatrixView<real const> matrix,
    ColumnVectorView<real const> in,
    ColumnVectorView<real> out) {
  MOCHI_ASSERT(matrix.Cols() == in.Rows(), "ApplyDense: dimension mismatch");
  MOCHI_ASSERT(matrix.Rows() == out.Rows(), "ApplyDense: dimension mismatch");
  out.SetZero();
  for (int j = 0; j < matrix.Cols(); ++j) {
    real const x = in[j];
    if (x == 0_r) {
      continue;
    }
    auto const col = matrix.Col(j);
    for (int i = 0; i < matrix.Rows(); ++i) {
      out[i] += col[i] * x;
    }
  }
}

// Use a Krylov solver for the linear problem dres * z = rhs, where dres is approximated.
// The approximate Hessian hat(dres) is used as preconditioner, and dres * v products
// are computed via finite differences in GetHessianVectorProduct.
static void KrylovSolveZ(
    entt::registry& reg,
    entt::entity island,
    SnleProblem<real>& problemForward,
    ColumnVectorView<real const> rhs,
    ColumnVectorView<real> outZ) {
  auto const& islandDofInfo = reg.get<CIslandDofInfo>(island);
  int const numDofs = islandDofInfo.dofsSize;

  // Retrieve backpropagation solver stats component.
  auto& islandBackPropSolverStats = reg.get<CIslandBackPropSolverStats>(island);

  // Handle trivial case
  if (IsZero(rhs)) {
    outZ.SetZero();
    islandBackPropSolverStats.stats = StageSolverStats{};
    return;
  }

  // Get outer and inner solver parameters
  auto const& backpropParams = reg.ctx<CBackPropagationSolverParams const>();
  auto const& simParams = reg.ctx<CSimulationParams const>();
  NewtonSolverParams newtonParamsForward;
  GetIslandNewtonParams(numDofs, simParams, newtonParamsForward);
  KrylovSolverParams& innerLParams = newtonParamsForward.lParams;
  innerLParams.absTol = backpropParams.innerSolverAbsTol;

  // Analytic outer operator: assemble the exact Hessian once, before the PSD
  // preconditioner assembly below reuses the problem's dresidual storage.
  bool const useAnalyticHvp = backpropParams.useAnalyticHvp;
  Matrix<real> exactHessian;
  if (useAnalyticHvp) {
    exactHessian = AssembleExactHessian(reg, island, problemForward);
  }

  // Assemble approximate Hessian hat(dres) for preconditioning
  AssemblyParams paramsDRes = {
      .assemObj = false, .assemRes = false, .assemDRes = true, .psdDRes = true};
  solver::AssembleIslandPipeline(reg, island, paramsDRes, problemForward);

  // Get the approximate Hessian matrix
  auto const& approxHessian = ToMatrix(problemForward.GetDResidual());

  // Create linear solver for the preconditioner (inner solver) using scene's linear solver settings
  auto& preconditionerRecyclingMgr = reg.get<CIslandPreconditioner>(island);
  LinearSolver<real> precLinearSolver(innerLParams, preconditionerRecyclingMgr);

  // Create callable operator for matrix-vector product via finite differences
  // This is used by krylov::Apply which calls A(v, Av) for non-matrix types
  auto hessianOp = [&](ColumnVectorView<real const> in, ColumnVectorView<real> out) {
    if (useAnalyticHvp) {
      ApplyDense(AsConstView(exactHessian), in, out);
      return;
    }
    GetHessianVectorProduct(reg, island, GradTarget::Current, problemForward, in, out);
  };

  // Create callable preconditioner: solves hat(dres) * z = r using the inner linear solver
  auto precOp = [&](ColumnVectorView<real const> in, ColumnVectorView<real> out) {
    out.SetZero();
    precLinearSolver.Solve(approxHessian, in, out, /*hasOperatorChanged*/ false);
  };

  // Initialize solution
  outZ.SetZero();

  // Try PCG first (faster for SPD systems), fall back to MINRES if non-SPD detected
  // PCG stopping criterion
  krylov::StatusResidualL2<krylov::UsualDot, real> pcgStatusCheck(
      backpropParams.outerSolverRelTol, // relTol: convergence if ||r|| <= relTol * ||rhs||
      backpropParams.outerSolverAbsTol, // absTol: convergence if ||r|| <= absTol
      static_cast<real>(newtonParamsForward.relDivTol));

  auto outerResult = krylov::PCG(
      hessianOp,
      rhs,
      outZ,
      precOp,
      backpropParams.outerSolverMaxIter,
      pcgStatusCheck,
      /*abortIfNotSpd*/ true,
      backpropParams.verbosity);

  // If PCG aborted due to non-SPD, fall back to MINRES
  if (!outerResult.converged && outerResult.numIterDone < backpropParams.outerSolverMaxIter) {
    if (backpropParams.verbosity >= VerbosityLevel::Warning) {
      MOCHI_LOG_WARNING(
          "PCG aborted at iteration %d (likely non-SPD). Falling back to MINRES.",
          outerResult.numIterDone);
    }

    // Reset solution for MINRES.
    // Note: An alternative could be to reuse the PCG solution as a warm start, by solving dres dz =
    // rhs_new, with rhs_new = rhs - dres z_pcg, and then z = z_pcg + dz. However, this could be a
    // bad idea if PCG progressed in a wrong direction.
    outZ.SetZero();

    // MINRES stopping criterion
    krylov::StatusImplicitResidualNorm<real> minresStatusCheck(
        backpropParams.outerSolverRelTol,
        backpropParams.outerSolverAbsTol,
        static_cast<real>(newtonParamsForward.relDivTol));

    outerResult = krylov::MinRes(
        hessianOp,
        rhs,
        outZ,
        precOp,
        backpropParams.outerSolverMaxIter,
        minresStatusCheck,
        backpropParams.verbosity);

    if (backpropParams.verbosity >= VerbosityLevel::Verbose) {
      MOCHI_LOG(
          "MINRES: Finished after %d iterations, final resNorm = %f, converged = %d",
          outerResult.numIterDone,
          outerResult.residualNorm,
          outerResult.converged);
    }

    if (backpropParams.verbosity >= VerbosityLevel::Warning && backpropParams.validateFiniteDiff) {
      // Diagnostic: compare MINRES's implicit residual (tracked internally via Givens rotations,
      // in the P^-1-norm) against the true residual computed from a fresh Hv product. A large
      // gap between them means MINRES's claimed convergence cannot be trusted. Two common
      // causes in this setting:
      //   1. FD noise on H*v corrupts the Lanczos recursion (residual-gap phenomenon). One bad
      //      Hv product can drive the implicit |eta| to a spurious near-zero via cancellations
      //      in the Givens rotations, causing MINRES to exit with a bad iterate.
      //   2. H is rank-deficient and rhs has a non-trivial component in null(H). Then Hz = rhs
      //      has no solution and MINRES wanders along null directions; the implicit norm drops
      //      but the true residual is bounded below by ||rhs_null||.
      MOCHI_FILO_STACK_ALLOCATOR(allocator, 2 * 256 * sizeof(real));
      ColumnVector<real> trueRes(rhs.Rows(), &allocator);
      hessianOp(AsConstView(outZ), AsView(trueRes)); // trueRes = H*outZ
      trueRes -= rhs; // trueRes = H*outZ - rhs (sign doesn't affect L2/P^-1 norms below)
      ColumnVector<real> pInvTrueRes(rhs.Rows(), &allocator);
      precOp(AsConstView(trueRes), AsView(pInvTrueRes));
      real const trueResNormPInv = Sqrt(Max(0_r, trueRes.Dot(pInvTrueRes)));
      real constexpr kRatioThreshold = 1e2_r;
      if (trueResNormPInv > kRatioThreshold * outerResult.residualNorm) {
        MOCHI_LOG_WARNING(
            "MINRES integrity check: implicit resNorm (P^-1) = %e, true resNorm (P^-1) = %e.",
            outerResult.residualNorm,
            static_cast<double>(trueResNormPInv));
      }
    }
  } else if (backpropParams.verbosity >= VerbosityLevel::Verbose) {
    MOCHI_LOG(
        "PCG: Finished after %d iterations, final resNorm = %f, converged = %d",
        outerResult.numIterDone,
        outerResult.residualNorm,
        outerResult.converged);
  }

  // With both flags set, cross-check the analytic operator against one central
  // finite difference at the solution; mismatches are reported through the same
  // finiteDiffValid diagnostic the FD path uses for its epsilon-robustness check.
  if (useAnalyticHvp && backpropParams.validateFiniteDiff && !IsZero(outZ)) {
    MOCHI_FILO_STACK_ALLOCATOR(validationAllocator, 2 * 256 * sizeof(real));
    ColumnVector<real> analyticHz(rhs.Rows(), &validationAllocator);
    ColumnVector<real> fdHz(rhs.Rows(), &validationAllocator);
    ApplyDense(AsConstView(exactHessian), AsConstView(outZ), AsView(analyticHz));
    GetHessianVectorProduct(
        reg, island, GradTarget::Current, problemForward, AsConstView(outZ), AsView(fdHz));
    fdHz -= analyticHz;
    real constexpr kAnalyticVsFdTol = 1e-2_r;
    real const relError =
        fdHz.Norm() / (analyticHz.Norm() + std::numeric_limits<real>::min());
    if (relError > kAnalyticVsFdTol) {
      islandBackPropSolverStats.finiteDiffValid = false;
      if (backpropParams.verbosity >= VerbosityLevel::Warning) {
        MOCHI_LOG_WARNING(
            "Analytic Hvp vs finite-difference cross-check: rel error = %e",
            static_cast<double>(relError));
      }
    }
  }

  // With validation on, replace the solver's residual estimate by the true residual of the
  // returned solution (a fresh Hessian-vector product: MINRES's implicit residual can be far
  // from it, see the integrity check above) and probe the symmetry of the operator. The adjoint
  // solve assumes H = H^T (the residual is the gradient of one merit function), which both PCG
  // and MINRES rely on; the probe compares rhs.(H z) with z.(H rhs), which agree for a symmetric
  // H up to the finite-difference noise of the products. Two extra products per solve.
  if (backpropParams.validateFiniteDiff) {
    MOCHI_FILO_STACK_ALLOCATOR(probeAllocator, 2 * 256 * sizeof(real));
    ColumnVector<real> hz(rhs.Rows(), &probeAllocator);
    hessianOp(AsConstView(outZ), AsView(hz));
    real const rhsDotHz = hz.Dot(rhs);
    hz -= rhs;
    outerResult.residualNorm = static_cast<double>(hz.Norm());
    ColumnVector<real> hRhs(rhs.Rows(), &probeAllocator);
    hessianOp(rhs, AsView(hRhs));
    real const zDotHRhs = hRhs.Dot(outZ);
    real const scale =
        0.5_r * (std::abs(rhsDotHz) + std::abs(zDotHRhs)) + std::numeric_limits<real>::min();
    real const asymmetry = std::abs(rhsDotHz - zDotHRhs) / scale;
    islandBackPropSolverStats.hessianAsymmetry = static_cast<double>(asymmetry);
    // Above the finite-difference noise of the products (measured 1.3e-4 at the default
    // epsilon 1e-8 on an articulated-vs-rigid frictional island, 1.3e-3 at 1e-7).
    real constexpr kAsymmetryWarnTol = 1e-2_r;
    if (asymmetry > kAsymmetryWarnTol && backpropParams.verbosity >= VerbosityLevel::Warning) {
      MOCHI_LOG_WARNING(
          "Adjoint operator symmetry probe: relative asymmetry %e. The step Jacobian is not symmetric, so the symmetric adjoint solve is only approximate.",
          static_cast<double>(asymmetry));
    }
  }

  // Record statistics
  StageSolverStats stats;
  stats.numIterDone = outerResult.numIterDone;
  stats.resNorm = static_cast<real>(outerResult.residualNorm);
  stats.resNormError = 0_r;
  stats.numLSIterDone = 0;
  islandBackPropSolverStats.stats = stats;
}

// Use a NR solver for the linear problem dres * z = rhs, where dres is approximated.
static void NewtonSolveZ(
    entt::registry& reg,
    entt::entity island,
    SnleProblem<real>& problemForward,
    ColumnVectorView<real const> rhs,
    ColumnVectorView<real> outZ) {
  auto const& islandDofInfo = reg.get<CIslandDofInfo>(island);
  int const numDofs = islandDofInfo.dofsSize;

  // Retrieve backpropagation solver stats component.
  auto& islandBackPropSolverStats = reg.get<CIslandBackPropSolverStats>(island);

  // Configure NR solver
  // ------------------------------------------------------------------------
  auto const& simParams = reg.ctx<CSimulationParams const>();
  NewtonSolverParams newtonParams;
  GetIslandNewtonParams(numDofs, simParams, newtonParams);
  // Apply general back-propagation parameters
  // - Currently, the back-propagation solver uses regular residual-based line search, but in the
  // test examples line-search never acted in practice.
  // - Allow reuse of the dresidual on all Newton iterations.
  newtonParams.dResidualAssemblyPeriod = std::numeric_limits<int>::max();
  // - Disable explosion control for two reasons: (1) The state was generated in a forward solve,
  // where possible explosions were already handled. (2) In the remote chance of an explosion, there
  // is no explosion handling mechanism, as the solver doesn't go through the actors.
  newtonParams.explosionControl = false;
  // Apply user-defined parameters for outer solver (Newton)
  auto const& solverParams = reg.ctx<CBackPropagationSolverParams const>();
  newtonParams.verbosity = solverParams.verbosity;
  newtonParams.maxIter = solverParams.outerSolverMaxIter;
  newtonParams.absTolRes = solverParams.outerSolverAbsTol;
  newtonParams.relTolRes = solverParams.outerSolverRelTol;
  newtonParams.convergenceMode = solverParams.outerSolverConvergenceMode;
  // Apply user-defined parameters for inner solver (linear)
  newtonParams.lParams.absTol = solverParams.innerSolverAbsTol;
  auto& preconditioner = reg.get<CIslandPreconditioner>(island);
  NewtonSolver<real> solver(newtonParams, preconditioner);

  // SNLE Callback Functions
  // -----------------------------------------------------------------------
  SnleProblemFunctions<real> functions;
  functions.onPostNewSolution = [](auto& /*problem*/) {}; // Not currently used.
  functions.onPostNewIncrement = [](auto& problem) { problem.solution += problem.increment; };
  functions.assemble = [&](SnleProblem<real>& problem, AssemblyParams const& params) {
    // dresidual assembly (only once)
    if (params.assemDRes) {
      // Assemble approx dresidual the first time
      AssemblyParams paramsDRes = {
          .assemObj = false,
          .assemRes = false,
          .assemDRes = true,
          .psdDRes = params.psdDRes,
          .fittedSaturationHessian = params.fittedSaturationHessian};
      solver::AssembleIslandPipeline(reg, island, paramsDRes, problem);
    }

    // residual or objective assembly
    if (params.assemObj || params.assemRes) {
      // Stack memory for 100 dofs
      MOCHI_FILO_STACK_ALLOCATOR(allocator, 100 * sizeof(real));

      // Finite-difference approximation of dres * z
      ColumnVector<real> dresTimesZ(problem.GetDofsSize(), &allocator);
      GetHessianVectorProduct(
          reg, island, GradTarget::Current, problemForward, problem.GetSolution(), dresTimesZ);

      if (params.assemObj) {
        // obj = 1/2 * zT * dres * z - zT * rhs
        auto const& z = problem.GetSolution();
        problem.objective = 0.5_r * z.Dot(dresTimesZ) - z.Dot(rhs);
      }

      if (params.assemRes) {
        // res = dres * z - rhs
        ColumnVector<real> res = std::move(dresTimesZ); // Reuse memory
        res -= rhs;

        // Write the actor residuals and convergence weights.
        auto descendants = reg.get<CIslandDescendants const>(island).actors;
        problem.actorResiduals.clear();
        problem.actorConvergenceWeights.clear();
        problem.actorResiduals.reserve(descendants.size());
        problem.actorConvergenceWeights.reserve(descendants.size());
        ecs::InvokeForEach<ecs::policy::AllowMutableExternalParams>(
            WriteToActorResidual, reg, descendants, AsConstView(res), std::ref(problem));
      }
    }
  };

  // SNLE Problem
  // -----------------------------------------------------------------------
  SnleProblem<real> problem(numDofs, numDofs, std::move(functions));
  problem.solution.SetZero();

  // Solve
  auto status = solver.Solve(problem);
  islandBackPropSolverStats.stats = StageSolverStats::FromNewtonSolverStatus(status);
  outZ = problem.GetSolution();
}

static void SetForceContainer(
    ColumnVectorView<real const> data,
    CActorDofInfo const& size,
    CDofOffset const& offset,
    CDiffForceGrad& outContainer) {
  outContainer = data.MiddleRows(offset.dofsOffset, size.dofsSize);
}

static void PrepareBackPropagationIslandAsync(
    entt::registry& reg,
    entt::entity island,
    CIslandDescendants const& descendants) {
  MOCHI_PROFILE_SCOPE();

  // The island pre-step operation is also needed for back-propagation
  PreStepIslandAsync(reg, descendants);

  // Define integration settings for this island
  auto const& simParams = reg.ctx<CSimulationParams const>();
  MOCHI_ASSERT(
      simParams.integrationMethod == IntegrationMethod::BackwardEuler,
      "Only Backward Euler is supported")
  auto const integrationParams =
      solver::CreateIslandTimeIntegrationParams(reg, descendants, simParams.integrationMethod);
  MOCHI_ASSERT(integrationParams.numStages == 1);
  solver::SetTimeIntegratorState(reg, descendants.actors, integrationParams, /*iStage*/ 0);

  // General settings and SNLE forward problem for contact assembly
  auto const& islandDofInfo = reg.get<CIslandDofInfo const>(island);
  SnleProblemFunctions<real> functions;
  SnleProblem<real> problem(islandDofInfo.dofsSize, islandDofInfo.poseSize, std::move(functions));

  // Set the stage-start state
  solver::PreFirstStageLocalPipeline(reg, descendants);
  solver::PreStageLocalPipeline(reg, descendants, problem);

  // Run collision detection if some actor has queries enabled
  if (std::any_of(descendants.actors.begin(), descendants.actors.end(), [&](auto const& e) {
        return reg.any_of<CQueryActorContactForces>(e);
      })) {
    // We must visit the full island, to account for cases where the queried actors act as
    // colliders.
    solver::UpdateDerivedStateBeforeAssembly(reg, GradTarget::Current, descendants);
  }
}

void mochi::PrepareBackPropagation(entt::registry& reg) {
  MOCHI_PROFILE_SCOPE();

  TaskSemaphore eachTask;
  reg.view<CIslandDescendants const>().each(
      [&](entt::entity island, CIslandDescendants const& descendants) {
        Schedule(eachTask, "PrepareBackPropagationIslandAsync", [&, island]() {
          PrepareBackPropagationIslandAsync(reg, island, descendants);
        });
      });
  eachTask.Wait();

  ecs::InvokeForEachGlobal(&PrepareContactForceAdjoints, reg);
}

static void BackPropagationSolveIslandAsync(
    entt::registry& reg,
    entt::entity island,
    CIslandDescendants const& descendants) {
  MOCHI_PROFILE_SCOPE();

  // General settings and SNLE forward problem for gradient assembly
  auto const& islandDofInfo = reg.get<CIslandDofInfo const>(island);
  int const solutionSize = islandDofInfo.poseSize;
  int const dofsSize = islandDofInfo.dofsSize;
  int const derivedDofsSize = reg.get<CIslandDerivedStateInfo const>(island).dofsSize;
  SnleProblemFunctions<real> functions; // dummy functions
  SnleProblem<real> problemForward(dofsSize, solutionSize, std::move(functions));

  // The state pair and contact data were prepared by PrepareBackPropagation(). Initialize
  // the local solution vector from the prepared current actor states without rerunning preparation.
  GetSolutions(problemForward.solution, reg, descendants.actors, /*baseOffset*/ 0);

  // Stack memory for 3 vectors of max size. The filo stack allocator requires that vectors are
  // deallocated in reverse order, so we allocate all at max size upfront.
  int const maxVecSize = Max(dofsSize, derivedDofsSize);
  MOCHI_FILO_STACK_ALLOCATOR(allocator, 3 * 256 * sizeof(real));
  ColumnVector<real> vec0(maxVecSize, &allocator);
  ColumnVector<real> vec1(maxVecSize, &allocator);
  ColumnVector<real> vec2(maxVecSize, &allocator);

  // 1. Collect rhs from CDiffContainerState
  auto rhs = vec0.TopRows(dofsSize); // use vec0 as rhs
  ecs::InvokeForEach<ecs::policy::AllowMutableExternalParams>(
      &GetContainer<GradTarget::Current>, reg, descendants.actors, rhs);

  // 2. Solve linear problem
  rhs *= -1_r;
  auto zSolve = vec1.TopRows(dofsSize); // use vec1 as zSolve
  auto const& solverParams = reg.ctx<CBackPropagationSolverParams const>();
  if (solverParams.useNewtonOuterSolver) {
    // Use Newton-Raphson outer solver
    NewtonSolveZ(reg, island, problemForward, rhs, zSolve);
  } else {
    // Use Krylov outer solver (PCG with MINRES fallback)
    KrylovSolveZ(reg, island, problemForward, rhs, zSolve);
  }

  // 3. Update actor gradient containers.
  auto forceGrad = vec0.TopRows(dofsSize); // use vec0 as grad; rhs is no longer needed
  forceGrad = -1_r * zSolve;
  ecs::InvokeForEach(&SetForceContainer, reg, descendants.actors, AsConstView(forceGrad));
  auto oldStateGrad =
      vec0.TopRows(dofsSize); // use vec0 as oldStateGrad; forceGrad is no longer needed
  GetHessianVectorProduct(reg, island, GradTarget::Previous, problemForward, zSolve, oldStateGrad);
  ecs::InvokeForEach(
      &SetContainer<GradTarget::Previous>, reg, descendants.actors, AsConstView(oldStateGrad));
  auto derivedStepGrad = vec0.TopRows(
      derivedDofsSize); // use vec0 as derivedStepGrad; oldStateGrad is no longer needed
  GetHessianVectorProduct(
      reg, island, GradTarget::PreviousDelta, problemForward, zSolve, derivedStepGrad);
  ecs::InvokeForEach(
      &SetContainer<GradTarget::PreviousDelta>,
      reg,
      descendants.actors,
      AsConstView(derivedStepGrad));

  // 4. Update input gradient containers if needed
  int const diffInputSize = reg.get<CIslandDiffInputInfo const>(island).size;
  if (diffInputSize > 0) {
    auto currentGrad =
        vec0.TopRows(diffInputSize); // use vec0 as currentGrad; derivedStepGrad is no longer needed
    auto previousGrad = vec2.TopRows(diffInputSize); // use vec2 as previousGrad
    GetHessianVectorProduct(
        reg, island, GradTarget::CurrentInput, problemForward, zSolve, currentGrad);
    GetHessianVectorProduct(
        reg, island, GradTarget::PreviousInput, problemForward, zSolve, previousGrad);
    auto const stepCounter = reg.ctx<CSceneStepCounter const>().value;
    ecs::InvokeForEach(
        UpdateTargetPoseGrad,
        reg,
        descendants.actors,
        AsConstView(currentGrad),
        AsConstView(previousGrad),
        stepCounter);
  }
}

bool mochi::IsSoftMaterialGradientSupported(CSoftMaterialParams const& material) {
  bool const isLame = std::visit(
      [](auto const& p) { return materials::kIsLameMaterial<std::decay_t<decltype(p)>>; },
      material.params);
  bool const homogeneous = std::visit(
      [](auto const& perElem) {
        if constexpr (std::is_same_v<std::decay_t<decltype(perElem)>, std::monostate>) {
          return false;
        } else {
          return perElem.size() == 1;
        }
      },
      material.perElementParams);
  return isLame && homogeneous;
}

// Read/write one of the differentiated soft material parameters (kSoftMaterialGrad* order) of
// the public parameter struct. Only valid for the Lame-type material models (see
// IsSoftMaterialGradientSupported).
static real& SoftMaterialField(SoftMaterialParams& params, int field) {
  auto lameField = [&](auto& typed) -> real& {
    return field == kSoftMaterialGradYoungsModulus ? typed.youngsModulus : typed.poissonRatio;
  };
  if (field == kSoftMaterialGradDensity) {
    return params.density;
  }
  if (field == kSoftMaterialGradMassDamping) {
    return params.massDampingCoefficient;
  }
  MOCHI_ASSERT(
      field == kSoftMaterialGradYoungsModulus || field == kSoftMaterialGradPoissonRatio,
      "Unexpected soft material gradient field");
  switch (params.type) {
    case SoftMaterialType::NeoHookean:
      return lameField(params.neoHookean);
    case SoftMaterialType::StVenantKirchhoff:
      return lameField(params.stVenantKirchhoff);
    case SoftMaterialType::LinearElastic:
      return lameField(params.linearElastic);
    default:
      MOCHI_ASSERT(false, "Soft material gradients cover Lame-type materials only");
      return params.density; // unreachable
  }
}

// Stack one actor's generalized-force adjoint into the island lambda vector.
// Invoked through ecs::InvokeForEach so entities without differentiability
// components (e.g. nested link actors in the descendants list) are skipped.
static void StackForceAdjoint(
    ColumnVectorView<real> outLambda,
    CActorDofInfo const& info,
    CDofOffset const& offset,
    CDiffForceGrad const& forceGrad) {
  outLambda.MiddleRows(offset.dofsOffset, info.dofsSize) = forceGrad;
}

// Accumulate this island's contribution to the parameter gradients: -lambda^T dR/dtheta
// by central finite differences of the assembled residual under a perturbed parameter,
// where lambda is the generalized-force adjoint (CDiffForceGrad) that
// BackPropagationSolveIslandAsync just computed. Runs after the island solves and
// sequentially across islands: it perturbs scene-global and per-entity parameter state,
// which parallel island tasks must not race on. Parameters covered: the scene gravity
// vector and, per contact-parameter owner (actors and nested links), the contact
// parameters listed in kNumContactParamGradients order.
static void AccumulateParameterGradientsIsland(
    entt::registry& reg,
    entt::entity island,
    CIslandDescendants const& descendants,
    Real3& outGradient) {
  auto const& islandDofInfo = reg.get<CIslandDofInfo const>(island);
  int const dofsSize = islandDofInfo.dofsSize;
  int const solutionSize = islandDofInfo.poseSize;
  if (dofsSize <= 0) {
    return;
  }

  MOCHI_FILO_STACK_ALLOCATOR(allocator, 4 * 256 * sizeof(real));

  // Record zero gradients up front for every contact-parameter owner in this island,
  // so a legitimately zero result is distinguishable from "never accumulated" (the
  // readout treats a missing component as a contract error).
  for (auto const e : descendants.actors) {
    if (reg.try_get<CContactParams const>(e) != nullptr &&
        reg.try_get<CDiffContactParamsGrad>(e) == nullptr) {
      reg.emplace<CDiffContactParamsGrad>(e);
    }
    if (reg.try_get<CRigidBodyInertia const>(e) != nullptr &&
        reg.try_get<CDiffDensityGrad>(e) == nullptr) {
      reg.emplace<CDiffDensityGrad>(e);
    }
  }
  for (auto const e : descendants.softActors) {
    auto const* material = reg.try_get<CSoftMaterialParams const>(e);
    if (material != nullptr && IsSoftMaterialGradientSupported(*material) &&
        reg.try_get<CDiffSoftMaterialGrad>(e) == nullptr) {
      reg.emplace<CDiffSoftMaterialGrad>(e);
    }
  }

  // Stack the per-actor force adjoints into one island vector. Entities in the
  // descendants list without the differentiability components contribute nothing.
  ColumnVector<real> lambda(dofsSize, &allocator);
  lambda.SetZero();
  ecs::InvokeForEach<ecs::policy::AllowMutableExternalParams>(
      &StackForceAdjoint, reg, descendants.actors, AsView(lambda));
  if (IsZero(lambda)) {
    return;
  }

  // Residual-only assembly problem at the prepared step states.
  SnleProblemFunctions<real> functions; // dummy; the assembly function is set below
  SnleProblem<real> problem(dofsSize, solutionSize, std::move(functions));
  // The caller (SceneImpl::BackPropagate) restored the exact step-state pair before
  // invoking the parameter stage, so the ECS holds the true, undrifted states here.
  GetSolutions(problem.solution, reg, descendants.actors, /*baseOffset*/ 0);
  AssemblyParams params = {.assemObj = false, .assemRes = true, .assemDRes = false};
  problem.SetAssemblyFunction([&](SnleProblem<real>& p, AssemblyParams const& /*unused*/) {
    solver::AssembleIslandPipeline(reg, island, params, p);
  });
  // A fresh SnleProblem's per-actor storage is created by its first dresidual
  // assembly (every existing user - KrylovSolveZ, the step-Jacobian path - starts
  // with one); residual-only assembly on an uninitialized problem crashes.
  AssemblyParams dresInit = {
      .assemObj = false, .assemRes = false, .assemDRes = true, .psdDRes = true};
  solver::AssembleIslandPipeline(reg, island, dresInit, problem);

  ColumnVector<real> delta(dofsSize, &allocator);
  delta.SetZero();
  ColumnVector<real> residualPlus(dofsSize, &allocator);
  ColumnVector<real> residualMinus(dofsSize, &allocator);
  auto evalResidual = [&](ColumnVectorView<real> out) {
    solver::PostNewIncrementLocalPipeline(reg, island, delta, problem.solution);
    problem.InvalidateCachedData();
    problem.UpdateResidual();
    out = problem.GetResidual();
  };
  // One-time re-anchor of the ECS state to the true solution (zero increment takes
  // the PostNewSolution path); the parameter perturbations below never move the
  // state, so subsequent evaluations stay anchored.
  solver::PostNewIncrementLocalPipeline(reg, island, delta, problem.solution);

  auto const& solverParams = reg.ctx<CBackPropagationSolverParams const>();

  // Gravity vector.
  auto& gravity = reg.ctx<CSceneGravity>();
  Vec4r const savedAccel = gravity.accel;
  Real3 const gravityRef = ToReal3(savedAccel);
  real const gravityEps = solverParams.epsFiniteDiff * (1_r + Sqrt(NormSqr(gravityRef)));
  for (int i = 0; i < 3; ++i) {
    Real3 gravityPerturbed = gravityRef;
    gravityPerturbed[i] = gravityRef[i] + gravityEps;
    gravity.accel = ToSimd(gravityPerturbed, 0_r);
    evalResidual(AsView(residualPlus));
    gravityPerturbed[i] = gravityRef[i] - gravityEps;
    gravity.accel = ToSimd(gravityPerturbed, 0_r);
    evalResidual(AsView(residualMinus));
    residualPlus -= residualMinus;
    outGradient[i] += -lambda.Dot(residualPlus) / (2_r * gravityEps);
  }
  gravity.accel = savedAccel;

  // Contact parameters, per owner (standalone actors and nested links alike). The
  // dissipative coefficients (Coulomb, viscous, normal damping) must be non-negative, and a
  // contact pair combines both owners' values by geometric mean (CombineContactParams). At a
  // zero-valued owner coefficient the loss is therefore a square-root cusp in that
  // coefficient when the partner's is positive (derivative +inf) and flat when it is zero;
  // a central difference would evaluate the negative side (Sqrt of a negative product). The
  // right-sided difference quotient at the finite-difference step is used there instead: it
  // is exactly zero for a zero partner and grows like 1 / Sqrt(step) otherwise - a finite,
  // correctly signed push for optimizers constrained to non-negative coefficients (see
  // test_diffsim_params.py, which pins both behaviors). The penalty coefficient is always
  // strictly positive.
  for (auto const e : descendants.actors) {
    auto* contactParams = reg.try_get<CContactParams>(e);
    if (contactParams == nullptr) {
      continue;
    }
    auto& outContactGrad = reg.get<CDiffContactParamsGrad>(e); // created above
    real* const fields[kNumContactParamGradients] = {
        &contactParams->penaltyCoefficient,
        &contactParams->coulombFrictionCoefficient,
        &contactParams->viscousFrictionCoefficient,
        &contactParams->normalViscousDampingCoefficient,
    };
    for (int f = 0; f < kNumContactParamGradients; ++f) {
      real const saved = *fields[f];
      real const eps = solverParams.epsFiniteDiff * (1_r + std::abs(saved));
      bool const rightSided = saved <= 0_r;
      *fields[f] = saved + eps;
      evalResidual(AsView(residualPlus));
      *fields[f] = rightSided ? saved : saved - eps;
      evalResidual(AsView(residualMinus));
      *fields[f] = saved;
      residualPlus -= residualMinus;
      outContactGrad.value[f] += -lambda.Dot(residualPlus) / (rightSided ? eps : 2_r * eps);
    }
  }

  // Density, per rigid-body-inertia owner (standalone rigid actors and articulated
  // links). SetDensity rescales mass and moment of inertia proportionally about the
  // inertia reference and restores bit-exactly (see RigidBodyInertia), which is what
  // makes this perturbation safe.
  for (auto const e : descendants.actors) {
    auto* inertia = reg.try_get<CRigidBodyInertia>(e);
    if (inertia == nullptr) {
      continue;
    }
    auto& outDensityGrad = reg.get<CDiffDensityGrad>(e); // created above
    real const density0 = inertia->GetDensity();
    real const eps = solverParams.epsFiniteDiff * density0; // relative; density > 0
    inertia->SetDensity(density0 + eps);
    evalResidual(AsView(residualPlus));
    inertia->SetDensity(density0 - eps);
    evalResidual(AsView(residualMinus));
    inertia->SetDensity(density0);
    residualPlus -= residualMinus;
    outDensityGrad.value += -lambda.Dot(residualPlus) / (2_r * eps);
  }

  // Soft material parameters, per standalone soft actor with a supported (homogeneous
  // Lame-type) material. Each perturbed parameter set goes through the forward setter's
  // conversion (soft::SetMaterialParams rebuilds the per-element Lame constants from Young's
  // modulus and Poisson's ratio; the residual reads density and mass damping directly), and
  // the component is restored bit-exactly from a copy afterwards. Mass damping is gated at
  // zero in the assembly (a non-positive coefficient disables the term), so at
  // massDampingCoefficient == 0 a central difference would straddle the gate and report half
  // the derivative; the right-sided difference is used there instead - the direction an
  // optimizer constrained to alpha >= 0 can move in.
  //
  // Step sizes: the residual is inertia-dominated (1/dt^2 scaling), so a difference over a
  // parameter that only moves the comparatively small elastic or damping terms amplifies
  // round-off by the ratio of the two - at epsFiniteDiff (1e-7) this was measured at 1e-4
  // relative error over an 8-step elastic oscillation. The Lame-type energies are homogeneous
  // of degree one in (lambda, mu) and hence linear in Young's modulus, and the inertia,
  // gravity and mass-damping terms are linear in density and in the damping coefficient, so
  // for those three a central difference has no truncation error at any step: a 1e-3
  // relative step removes the amplification. Poisson's ratio enters through
  // lambda = E nu / ((1 + nu)(1 - 2 nu)) and mu = E / (2 (1 + nu)), with poles at 0.5 and
  // -1: its step is 1e-2 of the distance to the nearer pole and the central difference is
  // Richardson-extrapolated (two steps, O(h^4)), which keeps the truncation error near 1e-8
  // for any admissible ratio while the step stays large enough against round-off.
  real constexpr kLinearParamRelativeStep = 1e-3_r;
  real constexpr kPoissonPoleFraction = 1e-2_r;
  for (auto const e : descendants.softActors) {
    auto* material = reg.try_get<CSoftMaterialParams>(e);
    if (material == nullptr || !IsSoftMaterialGradientSupported(*material)) {
      continue;
    }
    auto& outMaterialGrad = reg.get<CDiffSoftMaterialGrad>(e); // created above
    CSoftMaterialParams const saved = *material;
    SoftMaterialParams base;
    soft::GetMaterialParams(*material, base);
    auto evalAt = [&](int field, real value, ColumnVectorView<real> out) {
      SoftMaterialParams perturbed = base;
      SoftMaterialField(perturbed, field) = value;
      soft::SetMaterialParams(perturbed, *material);
      evalResidual(out);
    };
    // -lambda^T (R(value + h) - R(value - h)) / (2 h), or the right-sided variant.
    auto centralDifference = [&](int f, real value, real h, bool rightSided) {
      evalAt(f, value + h, AsView(residualPlus));
      evalAt(f, rightSided ? value : value - h, AsView(residualMinus));
      residualPlus -= residualMinus;
      return -lambda.Dot(residualPlus) / (rightSided ? h : 2_r * h);
    };
    for (int f = 0; f < kNumSoftMaterialParamGradients; ++f) {
      real const value = SoftMaterialField(base, f);
      if (f == kSoftMaterialGradPoissonRatio) {
        real const h = kPoissonPoleFraction * Min(0.5_r - value, 1_r + value);
        MOCHI_ASSERT(h > 0_r, "Poisson's ratio must lie in (-1, 0.5)");
        real const coarse = centralDifference(f, value, h, false);
        real const fine = centralDifference(f, value, 0.5_r * h, false);
        outMaterialGrad.value[f] += (4_r * fine - coarse) / 3_r;
        continue;
      }
      real const h = kLinearParamRelativeStep * (1_r + std::abs(value));
      bool const rightSided = (f == kSoftMaterialGradMassDamping && value <= 0_r);
      outMaterialGrad.value[f] += centralDifference(f, value, h, rightSided);
    }
    *material = saved;
  }
}

void mochi::AccumulateParameterGradients(entt::registry& reg) {
  MOCHI_PROFILE_SCOPE();
  if (reg.try_ctx<CDiffGravityGrad>() == nullptr) {
    // ResetBackPropagation has not run yet; the parameter readouts report this as an
    // error at read time rather than returning a silent zero.
    return;
  }
  Real3 gradient{};
  reg.view<CIslandDescendants const>().each(
      [&](entt::entity island, CIslandDescendants const& descendants) {
        AccumulateParameterGradientsIsland(reg, island, descendants, gradient);
      });
  reg.ctx<CDiffGravityGrad>().value += gradient;
}

void mochi::ResetContactParamsGradContainers(CDiffContactParamsGrad& outGrad) {
  outGrad.value.fill(0_r);
}

void mochi::ResetDensityGradContainers(CDiffDensityGrad& outGrad) {
  outGrad.value = 0_r;
}

void mochi::ResetSoftMaterialGradContainers(CDiffSoftMaterialGrad& outGrad) {
  outGrad.value.fill(0_r);
}

void mochi::BackPropagationSolve(entt::registry& reg) {
  MOCHI_PROFILE_SCOPE();
  // Emplace CIslandBackPropSolverStats for all islands.
  reg.view<TagIsland>().each(
      [&](entt::entity island) { reg.emplace_or_replace<CIslandBackPropSolverStats>(island); });

  TaskSemaphore eachTask;
  reg.view<CIslandDescendants const>().each(
      [&](entt::entity island, CIslandDescendants const& descendants) {
        Schedule(eachTask, "BackPropagateIsland", [&, island]() {
          BackPropagationSolveIslandAsync(reg, island, descendants);
        });
      });

  eachTask.Wait();
}

// Assign/Acquire to global Jacobian matrices, using CSceneStateOffset to determine block offset
static void AssignDResBlk(
    entt::registry& reg,
    Span<entt::entity const> actors,
    MatrixView<real> dRes,
    MatrixView<real const> dResBlk) {
  for (auto const& r : actors) {
    int rOffGlobal = reg.get<CSceneStateOffset const>(r).dofsOffset;
    int rOff = reg.get<CDofOffset const>(r).dofsOffset;
    int rDofs = reg.get<CActorDofInfo const>(r).dofsSize;
    for (auto const& c : actors) {
      int cOffGlobal = reg.get<CSceneStateOffset const>(c).dofsOffset;
      int cOff = reg.get<CDofOffset const>(c).dofsOffset;
      int cDofs = reg.get<CActorDofInfo const>(c).dofsSize;
      dRes.Block(rOffGlobal, cOffGlobal, rDofs, cDofs) = dResBlk.Block(rOff, cOff, rDofs, cDofs);
    }
  }
}

static void AcquireDResBlk(
    entt::registry& reg,
    Span<entt::entity const> actors,
    MatrixView<real const> dRes,
    MatrixView<real> dResBlk) {
  for (auto const& r : actors) {
    int rOffGlobal = reg.get<CSceneStateOffset const>(r).dofsOffset;
    int rOff = reg.get<CDofOffset const>(r).dofsOffset;
    int rDofs = reg.get<CActorDofInfo const>(r).dofsSize;
    for (auto const& c : actors) {
      int cOffGlobal = reg.get<CSceneStateOffset const>(c).dofsOffset;
      int cOff = reg.get<CDofOffset const>(c).dofsOffset;
      int cDofs = reg.get<CActorDofInfo const>(c).dofsSize;
      dResBlk.Block(rOff, cOff, rDofs, cDofs) = dRes.Block(rOffGlobal, cOffGlobal, rDofs, cDofs);
    }
  }
}

void mochi::ComputeHqx(
    int numIslandDofs,
    Span<entt::entity const> actors,
    CActorSnle const& actorSnle,
    CDofOffset const& dofOffset,
    CActorDerivedStateInfo const& derivedStateInfo,
    CForwardPropContainerDerivedStateJac& outDerivedState) {
  int cols = derivedStateInfo.dofsSize;
  outDerivedState.actors = actors;
  outDerivedState.numIslandDofs = numIslandDofs;
  outDerivedState.data.Resize(numIslandDofs, cols);

  // Fill the mixed Hessian H_{qx} = [df/dδ]
  // Note that [df/dδ] is a block diagonal matrix because this term is due to inertial term.
  // This is why we can store the corresponding columns of the derived state matrix in the per-actor
  // struct: CForwardPropContainerDerivedStateJac
  outDerivedState.data.SetZero();
  outDerivedState.data.MiddleRows(dofOffset.dofsOffset, derivedStateInfo.dofsSize) =
      actorSnle.UseReduced() ? ToMatrix(AsConstView(actorSnle.reducedDResidual))
                             : ToMatrix(AsConstView(actorSnle.fullDResidual));
}

void mochi::ComputeDqDDerivedState(
    int numIslandDofs,
    LU<real> const& invDRes,
    Span<entt::entity const> actors,
    CActorSnle const& actorSnle,
    CDofOffset const& dofOffset,
    CActorDerivedStateInfo const& derivedStateInfo,
    CForwardPropContainerDerivedStateJac& outDerivedState) {
  // Compute H_{qx} = [df/dδ]
  ComputeHqx(numIslandDofs, actors, actorSnle, dofOffset, derivedStateInfo, outDerivedState);

  // Solve for dqk/dδ = -[df/dqk]^{-1} * H_{qx}
  outDerivedState.data *= -1_r;
  invDRes.LeftSolveInPlace(outDerivedState.data);
}

static void ShiftDqDDerivedState(
    int numIslandDofs,
    MatrixView<real> outJacCurr,
    MatrixView<real> outJacOld,
    entt::entity e,
    entt::registry& reg,
    CDofOffset const& dofOffset,
    CActorDofInfo const& dofInfo,
    CActorDerivedStateInfo const& derivedStateInfo,
    CForwardPropContainerDerivedStateJac const& derivedState,
    CDiffContainerDerivedState& outDerivedState) {
  int cols = derivedStateInfo.dofsSize;
  MatrixView<real const> derivedStateMat(derivedState.data.data(), numIslandDofs, cols);

  auto jacCurrActor = outJacCurr.MiddleCols(dofOffset.dofsOffset, dofInfo.dofsSize);
  auto jacOldActor = outJacOld.MiddleCols(dofOffset.dofsOffset, dofInfo.dofsSize);
  // The data structure CForwardPropContainerDerivedStateJac stores the sub-matrix dqk/dδ of size:
  //   |numIslandDofs| x |#derived state of the actor|
  // The matrix is computed via:
  //   dqk/dδ = -[df/dqk]^{-1} [df/dδ]
  for (int row = 0; row < numIslandDofs; row++) {
    // CDiffContainerDerivedState = dqk_row/dδ
    outDerivedState = derivedStateMat.Row(row).Transpose();
    // CDiffContainerState = dqk_row/dq_k-1 = dqk_row/dδ * dδ/dq_k-1
    ecs::TryInvokeOnEntity(rigid::ProjectDerivedStateGradient, reg, e);
    ecs::TryInvokeOnEntity(rigid::ShiftDerivedStateGradient, reg, e);
    // dqk_row/dq_k-1 = dqk_row/dq_k-1 + dqk_row/dδ * dδ/dq_k-1
    jacCurrActor.Row(row) += reg.get<CDiffContainerState const>(e).Transpose();
    // CDiffContainerState = dqk_row/dq_k-2 = dqk_row/dδ * dδ/dq_k-2
    // In the case of rigid body, this is just an assignment
    ecs::TryInvokeOnEntity(rigid::ProjectDerivedStateGradient, reg, e);
    // dqk_row/dq_k-2 = dqk_row/dδ * dδ/dq_k-2
    jacOldActor.Row(row) = reg.get<CDiffContainerState const>(e).Transpose();
  }
}

static void StepJacobianSolveIslandAsync(
    entt::registry& reg,
    entt::entity island,
    CIslandMembers const& members,
    CIslandDescendants const& descendants,
    MatrixView<real> outJacCurr) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(!descendants.actors.empty(), "Empty islands should have been pruned");

  // General settings and SNLE forward problem for mixed Hessian assembly
  auto const& islandDofInfo = reg.get<CIslandDofInfo>(island);
  int const solutionSize = islandDofInfo.poseSize;
  int const dofsSize = islandDofInfo.dofsSize;

  // Initialize island assemble problem
  SnleProblemFunctions<real> functions;
  functions.onPostNewSolution = [](auto& /*problem*/) {}; // Not currently used.
  functions.onPostNewIncrement = [](auto& /*problem*/) {}; // Not currently used.
  functions.assemble = [&](SnleProblem<real>& problem, AssemblyParams const& params) {
    solver::AssembleIslandPipeline(reg, island, params, problem);
  };
  SnleProblem<real> problem(dofsSize, solutionSize, std::move(functions));

  // Set the stage-start state
  solver::PreFirstStageLocalPipeline(reg, descendants);
  solver::PreStageLocalPipeline(reg, descendants, problem);

  // Assemble problem with psdDRes=false, notifying exact Hessian
  AssemblyParams paramsExactDRes = {
      .assemObj = false, .assemRes = false, .assemDRes = true, .psdDRes = false};
  problem.UpdateObjResDRes(paramsExactDRes);

  // Direct factorization using LU
  LU<real> invDRes(ToMatrix(problem.GetDResidual()));

  // Derivative with respect to previous state
  paramsExactDRes.gradTarget = GradTarget::Previous;
  problem.SetAssemblyFunction([&](SnleProblem<real>& problem, AssemblyParams const& params) {
    mochi::solver::AssembleIslandPipeline(reg, island, params, problem);
  });
  problem.InvalidateCachedData();
  problem.UpdateObjResDRes(paramsExactDRes);
  // dqk/dqk-1 = -[d2f/dqk2]^{-1} * d2f/dqk/dqk-1
  auto jacCurrBlk = ToMatrix(problem.GetDResidual());
  jacCurrBlk *= -1_r;
  invDRes.LeftSolveInPlace(jacCurrBlk);
  AssignDResBlk(reg, members.actors, outJacCurr, jacCurrBlk);

  // Derivative with respect to delta
  paramsExactDRes.gradTarget = GradTarget::PreviousDelta;
  problem.SetAssemblyFunction([&](SnleProblem<real>& problem, AssemblyParams const& params) {
    mochi::solver::AssembleIslandPipeline(reg, island, params, problem);
  });
  problem.InvalidateCachedData();
  problem.UpdateObjResDRes(paramsExactDRes);
  // dqk/dδ = -[d2f/dqk2]^{-1} * d2f/dqk/dδ
  ecs::InvokeForEach(
      ComputeDqDDerivedState,
      reg,
      members.actors,
      dofsSize,
      std::cref(invDRes),
      MakeConstSpan(members.actors));
}

static void StepJacobianShiftActor(
    MatrixView<real> jacCurr,
    MatrixView<real> jacOld,
    entt::registry& reg,
    CForwardPropContainerDerivedStateJac const& derivedState) {
  MOCHI_PROFILE_SCOPE();

  // Initialize jacobian blocks to zero
  Matrix<real> jacCurrBlk(derivedState.numIslandDofs, derivedState.numIslandDofs);
  Matrix<real> jacOldBlk(derivedState.numIslandDofs, derivedState.numIslandDofs);
  AcquireDResBlk(reg, derivedState.actors, jacCurr, jacCurrBlk);

  // Assuming that we have computed CForwardPropContainerDerivedStateJac, which stores dqk/dδ
  // The following call propagates Jacobian to compute:
  //   dqk/dqk-1 += dqk/dδ * dδ/dqk-1
  //   dqk/dqk-2  = dqk/dδ * dδ/dqk-2
  // This is same computation as BackPropagation, but applied row-by-row, i.e.:
  // We extract dqk_i/dδ and store this vector into CDiffContainerDerivedState
  // We can then use the backpropagation functionality to compute:
  //   dqk_i/dqk-1 += dqk_i/dδ * dδ/dqk-1
  //   dqk_i/dqk-2  = dqk_i/dδ * dδ/dqk-2
  ecs::InvokeForEach<ecs::policy::AllowFullRegistryAccess>(
      ShiftDqDDerivedState,
      reg,
      derivedState.actors,
      derivedState.numIslandDofs,
      AsView(jacCurrBlk),
      AsView(jacOldBlk));

  // assign to scene-wise global jacobian matrix
  AssignDResBlk(reg, derivedState.actors, jacCurr, jacCurrBlk);
  AssignDResBlk(reg, derivedState.actors, jacOld, jacOldBlk);
}

void mochi::StepJacobianSolve(entt::registry& reg, MatrixView<real> jacCurr) {
  MOCHI_PROFILE_SCOPE();

  // Initialize Jacobian matrix to zero
  jacCurr.SetZero();

  reg.view<CIslandMembers const, CIslandDescendants const>().each(
      [&](entt::entity island,
          CIslandMembers const& members,
          CIslandDescendants const& descendants) {
        // The island pre-step operation is also needed for forward-propagation
        PreStepIslandAsync(reg, descendants);
        // Now forward-propagate each island
        StepJacobianSolveIslandAsync(reg, island, members, descendants, jacCurr);
        // The island post-step operation is not needed for forward-propagation
      });
}

void mochi::StepJacobianShiftAndProject(
    entt::registry& reg,
    MatrixView<real> outJacCurr,
    MatrixView<real> outJacOld) {
  MOCHI_PROFILE_SCOPE();
  // Initialize Jacobian matrix to zero
  outJacOld.SetZero();

  // Shift each stateNew island exactly once, keyed on the stored grouping (the actors list each
  // actor recorded in ComputeHqx), not the current CIslandMembers: GetStepJacobian has since
  // restored stateCurr/stateOld, so current islanding can differ from stateNew's. De-duplicate by
  // the island's first member so a split/merged current island can't double- or under-apply it.
  std::unordered_set<entt::id_type> processedIslands;
  reg.view<CForwardPropContainerDerivedStateJac const>().each(
      [&](entt::entity /*e*/, CForwardPropContainerDerivedStateJac const& derivedState) {
        if (derivedState.actors.empty()) {
          return;
        }
        if (processedIslands.insert(static_cast<entt::id_type>(derivedState.actors[0])).second) {
          StepJacobianShiftActor(AsView(outJacCurr), AsView(outJacOld), reg, derivedState);
        }
      });
}

void mochi::EmplaceDifferentiabilityComponents(
    int numDerivedStateDofs,
    entt::registry& reg,
    entt::entity e,
    CActorDofInfo const& dofInfo) {
  // Emplace components for derived-state vector indexing
  auto& derivedStateInfo = reg.emplace_or_replace<CActorDerivedStateInfo>(e);
  derivedStateInfo.dofsSize = numDerivedStateDofs;
  reg.emplace_or_replace<CDerivedStateOffset>(e);

  // Emplace components for differentiable-input vector indexing
  auto& diffInputInfo = reg.emplace_or_replace<CActorDiffInputInfo>(e);
  diffInputInfo.dofsSize = 0;
  reg.emplace_or_replace<CDiffInputOffset>(e);

  // Emplace components to store temporary gradient data during the forward/back-propagation step.
  reg.emplace<CDiffContainerState>(e, dofInfo.dofsSize);
  reg.emplace<CDiffContainerDerivedState>(e, numDerivedStateDofs);
  reg.emplace<CDiffForceGrad>(e, dofInfo.dofsSize);
  reg.emplace<CForwardPropContainerDerivedStateJac>(e);

  // Emplace components to store adjoints per-entity across BackPropagate calls.
  reg.emplace<CDiffStateGrad>(e, dofInfo.dofsSize);
  reg.emplace<CDiffDerivedStepGrad>(e, numDerivedStateDofs);

  // Emplace actor data
  reg.emplace<CSceneStateOffset>(e);
}

void mochi::EmplaceDifferentiableContactComponents(
    entt::registry& reg,
    entt::entity e,
    CActorDofInfo const& dofInfo) {
  // Emplace components to store contact-force adjoints per-entity.
  reg.emplace<CDiffContactGrad<GradTarget::Current>>(e, dofInfo.dofsSize);
  reg.emplace<CDiffContactGrad<GradTarget::Previous>>(e, dofInfo.dofsSize);
}

void mochi::EmplaceConstraintDifferentiabilityComponents(entt::registry& reg, entt::entity e) {
  auto const& info = reg.get<CConstraintInfo const>(e);
  MOCHI_ASSERT(
      GetNumConstrainedTargets(info.type) > 0,
      "A constraint with no target does not require differentiability components");
  MOCHI_ASSERT(
      info.GetNumTargets() == GetNumConstrainedTargets(info.type),
      "This constraint does not support differentiability of targets");
  MOCHI_ASSERT(
      !info.hasMixedLinks,
      "Differentiability not supported for constraints between links of an articulated body "
      "and external actors.");
  reg.emplace<TagConstraintWithDifferentiableInput>(e);
  reg.emplace<CConstraintGlobalInputSparsityCache>(e);
}

void mochi::ResetBackPropagationContainers(
    CDiffStateGrad& outGradState,
    CDiffDerivedStepGrad& outGradDerivedStep,
    CDiffTargetPoseGrad* outTargetPoseGrad) {
  outGradState.value.SetZero();
  outGradDerivedStep.value.SetZero();
  if (outTargetPoseGrad) {
    outTargetPoseGrad->propagated.SetZero();
  }
}

void mochi::PrepareContactForceAdjoints(
    CQueryActorContactForces const& /*queryActorContactForces*/,
    [[maybe_unused]] CRequiresFarSdfEvaluation const* farSdfEval,
    CActiveCollisions<ContactType::Async, TimeStep::Current>& outActiveCollisionsAsync,
    CActiveCollisions<ContactType::Sync, TimeStep::Current>& outActiveCollisionsSync,
    CCollJacs<CollRole::Collider>* outColliderJacs) {
  MOCHI_ASSERT_VERBOSE(
      !farSdfEval, "Far SDF evaluation is not compatible with contact-force queries.");

  auto prepareAdjoints = [](ContactDetectionResult& collisionResult) {
    auto& adjoints = collisionResult.forcePerUnitArea;
    adjoints.resize_noinit(collisionResult.sampleIndices.size());
    std::fill(adjoints.begin(), adjoints.end(), Real3{});
  };

  for (auto& collision : outActiveCollisionsAsync) {
    prepareAdjoints(collision.collisionResult);
  }
  for (auto& collision : outActiveCollisionsSync) {
    prepareAdjoints(collision.collisionResult);
  }
  if (outColliderJacs) {
    for (auto& jac : *outColliderJacs) {
      prepareAdjoints(*jac.query);
    }
  }
}

namespace mochi::differentiable {
void InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<TagDifferentiableScene>(reg);
  ecs::RegisterComponent<TagBackPropagationPrepared>(reg);
  ecs::RegisterComponent<CStatePair>(reg);
  ecs::RegisterComponent<TagConstraintWithDifferentiableInput>(reg);
  ecs::RegisterComponent<CBackPropagationSolverParams>(reg);
  ecs::RegisterComponent<CBackPropagationSceneStats>(reg);
  ecs::RegisterComponent<CIslandDerivedStateInfo>(reg);
  ecs::RegisterComponent<CActorDerivedStateInfo>(reg);
  ecs::RegisterComponent<CDerivedStateOffset>(reg);
  ecs::RegisterComponent<CIslandDiffInputInfo>(reg);
  ecs::RegisterComponent<CActorDiffInputInfo>(reg);
  ecs::RegisterComponent<CDiffInputOffset>(reg);
  ecs::RegisterComponent<CDiffContainerState>(reg);
  ecs::RegisterComponent<CDiffContainerDerivedState>(reg);
  ecs::RegisterComponent<CForwardPropContainerDerivedStateJac>(reg);
  ecs::RegisterComponent<CSceneStateOffset>(reg);
  ecs::RegisterComponent<CDiffForceGrad>(reg);
  ecs::RegisterComponent<CDiffTargetPoseGrad>(reg);
  ecs::RegisterComponent<CDiffStateGrad>(reg);
  ecs::RegisterComponent<CDiffDerivedStepGrad>(reg);
  ecs::RegisterComponent<CDiffContactGrad<GradTarget::Current>>(reg);
  ecs::RegisterComponent<CDiffContactGrad<GradTarget::Previous>>(reg);
  ecs::RegisterComponent<CIslandBackPropSolverStats>(reg);
}
} // namespace mochi::differentiable
