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
#include "mochi_rod.h"
#include "mochi_simulation.h"
#include "mochi_soft.h"
#include "mochi_solve.h"
#include "mochi_step.h"

#include <unordered_map>
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
  // External forces do not depend on the state and must not go through the chart transport
  // below (see AssemblyParams::assemExternalForces); the force-input gradient is read from the
  // adjoint solution directly.
  AssemblyParams params = {
      .assemObj = false,
      .assemRes = true,
      .assemDRes = false,
      .gradTarget = gradTarget,
      .assemExternalForces = false};
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
  auto evalHvpAtEps = [&](real stepScale, ColumnVectorView<real> outGrad) {
    delta = (stepScale * eps) * vector;
    evalGradient(outGrad);
    delta *= -1_r;
    evalGradient(auxGrad);
    outGrad -= auxGrad;
    outGrad *= (0.5_r / (stepScale * eps));
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

// The step residual is evaluated in the chart of the iterate itself (a left rotation increment at
// the current rotation), so the Jacobian of the implicit step map is the derivative of the
// moving-chart residual, J = d r(exp(delta) x) / d delta. The finite-difference products
// transport every residual back to the chart at x (rigid::TransportGradient) and the analytic
// assembly differentiates in that fixed chart, so both produce the symmetric Hessian H of the
// merit. J differs from H by the derivative of the chart acting on the state-independent part of
// the residual, which at a solved step equals the external torque tau of a standalone rigid actor:
// exp(eta) exp(delta) = exp(delta + eta - delta x eta / 2 + ...) gives J = H - 1/2 [tau]x on that
// actor's rotation block, and the adjoint solve needs J^T = H + 1/2 [tau]x. Without this term the
// torque and rotational gradients were off by half the per-step rotation the torque induces
// (2.2e-4 relative at 0.3 N m on a free cube, 2026-09-03).
//
// Adds sign * 1/2 tau x v_rot to out for every standalone rigid actor of the island with an
// external torque; returns whether there is one.
static bool AddRigidTorqueChartTerm(
    entt::registry const& reg,
    CIslandDescendants const& descendants,
    real sign,
    ColumnVectorView<real const> in,
    ColumnVectorView<real> out) {
  bool any = false;
  for (entt::entity const e : descendants.rigidActors) {
    if (reg.any_of<TagArticulatedLinkActor>(e)) {
      continue;
    }
    auto const* externalForces = reg.try_get<CExternalForces const>(e);
    if (!externalForces || externalForces->Empty()) {
      continue;
    }
    Real3 tau = {};
    bool hasTorque = false;
    for (int i = 0; i < isize(externalForces->dofs); ++i) {
      int const dof = externalForces->dofs[i];
      if (dof >= RigidSize::kDTrans && dof < RigidSize::kDAll && externalForces->forces[i] != 0_r) {
        tau[dof - RigidSize::kDTrans] = externalForces->forces[i];
        hasTorque = true;
      }
    }
    if (!hasTorque) {
      continue;
    }
    any = true;
    int const offset = reg.get<CDofOffset const>(e).dofsOffset + RigidSize::kDTrans;
    real const v0 = in[offset];
    real const v1 = in[offset + 1];
    real const v2 = in[offset + 2];
    real const half = 0.5_r * sign;
    out[offset] += half * (tau[1] * v2 - tau[2] * v1);
    out[offset + 1] += half * (tau[2] * v0 - tau[0] * v2);
    out[offset + 2] += half * (tau[0] * v1 - tau[1] * v0);
  }
  // The torque on a Free or Spherical joint enters through the same merit as a rigid torque,
  // evaluated on the joint transform's rotation step (AssembleExternalForces ->
  // AddRigidBodyExternalForces), so the joint's rotation block of the reduced step Jacobian
  // carries the same term.
  for (entt::entity const e : descendants.compoundActors) {
    auto const* externalForces = reg.try_get<CExternalForces const>(e);
    if (!externalForces || externalForces->Empty()) {
      continue;
    }
    auto const* joints = reg.get<CArticulatedBodyShape const>(e).shape->GetJointsData();
    int const dofsOffset = reg.get<CDofOffset const>(e).dofsOffset;
    for (int joint = 0; joint < isize(joints->jointTypes); ++joint) {
      if (joints->jointTypes[joint] != ArticulatedJointType::Free &&
          joints->jointTypes[joint] != ArticulatedJointType::Spherical) {
        continue;
      }
      int const rotOffset = joints->dofInfo[joint].GetRotOffset();
      Real3 tau = {};
      bool hasTorque = false;
      for (int i = 0; i < isize(externalForces->dofs); ++i) {
        int const dof = externalForces->dofs[i];
        if (dof >= rotOffset && dof < rotOffset + RigidSize::kDRot &&
            externalForces->forces[i] != 0_r) {
          tau[dof - rotOffset] = externalForces->forces[i];
          hasTorque = true;
        }
      }
      if (!hasTorque) {
        continue;
      }
      any = true;
      int const offset = dofsOffset + rotOffset;
      real const v0 = in[offset];
      real const v1 = in[offset + 1];
      real const v2 = in[offset + 2];
      real const half = 0.5_r * sign;
      out[offset] += half * (tau[1] * v2 - tau[2] * v1);
      out[offset + 1] += half * (tau[2] * v0 - tau[0] * v2);
      out[offset + 2] += half * (tau[0] * v1 - tau[1] * v0);
    }
  }
  return any;
}

// The dense counterpart: adds sign * 1/2 [tau]x to the rotation diagonal block of every standalone
// rigid actor with an external torque (sign -1 turns the fixed-chart Hessian into J).
static void AddRigidTorqueChartTerm(
    entt::registry const& reg, CIslandDescendants const& descendants, real sign, Matrix<real>& mat) {
  for (entt::entity const e : descendants.rigidActors) {
    if (reg.any_of<TagArticulatedLinkActor>(e)) {
      continue;
    }
    auto const* externalForces = reg.try_get<CExternalForces const>(e);
    if (!externalForces || externalForces->Empty()) {
      continue;
    }
    Real3 tau = {};
    for (int i = 0; i < isize(externalForces->dofs); ++i) {
      int const dof = externalForces->dofs[i];
      if (dof >= RigidSize::kDTrans && dof < RigidSize::kDAll) {
        tau[dof - RigidSize::kDTrans] = externalForces->forces[i];
      }
    }
    int const o = reg.get<CDofOffset const>(e).dofsOffset + RigidSize::kDTrans;
    real const half = 0.5_r * sign;
    mat(o, o + 1) += -half * tau[2];
    mat(o, o + 2) += half * tau[1];
    mat(o + 1, o) += half * tau[2];
    mat(o + 1, o + 2) += -half * tau[0];
    mat(o + 2, o) += -half * tau[1];
    mat(o + 2, o + 1) += half * tau[0];
  }
  // The same term on the rotation block of every Free or Spherical joint with a torque.
  for (entt::entity const e : descendants.compoundActors) {
    auto const* externalForces = reg.try_get<CExternalForces const>(e);
    if (!externalForces || externalForces->Empty()) {
      continue;
    }
    auto const* joints = reg.get<CArticulatedBodyShape const>(e).shape->GetJointsData();
    int const dofsOffset = reg.get<CDofOffset const>(e).dofsOffset;
    for (int joint = 0; joint < isize(joints->jointTypes); ++joint) {
      if (joints->jointTypes[joint] != ArticulatedJointType::Free &&
          joints->jointTypes[joint] != ArticulatedJointType::Spherical) {
        continue;
      }
      int const rotOffset = joints->dofInfo[joint].GetRotOffset();
      Real3 tau = {};
      for (int i = 0; i < isize(externalForces->dofs); ++i) {
        int const dof = externalForces->dofs[i];
        if (dof >= rotOffset && dof < rotOffset + RigidSize::kDRot) {
          tau[dof - rotOffset] = externalForces->forces[i];
        }
      }
      int const o = dofsOffset + rotOffset;
      real const half = 0.5_r * sign;
      mat(o, o + 1) += -half * tau[2];
      mat(o, o + 2) += half * tau[1];
      mat(o + 1, o) += half * tau[2];
      mat(o + 1, o + 2) += -half * tau[0];
      mat(o + 2, o) += -half * tau[1];
      mat(o + 2, o + 1) += half * tau[0];
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
  islandBackPropSolverStats.usedMinresFallback = false;

  // Handle trivial case
  if (IsZero(rhs)) {
    outZ.SetZero();
    islandBackPropSolverStats.stats = StageSolverStats{};
    islandBackPropSolverStats.converged = true;
    islandBackPropSolverStats.residualThreshold = 0.0;
    return;
  }

  // Get outer and inner solver parameters
  auto const& backpropParams = reg.ctx<CBackPropagationSolverParams const>();
  auto const& simParams = reg.ctx<CSimulationParams const>();
  NewtonSolverParams newtonParamsForward;
  GetIslandNewtonParams(numDofs, simParams, newtonParamsForward);
  KrylovSolverParams& innerLParams = newtonParamsForward.lParams;
  // The preconditioner solve must be judged in the outer solver's norm (plain residual L2)
  // and to a tolerance below the outer stopping threshold: judged in the scene default
  // (the preconditioned residual norm) with an absolute floor, it returned a zero
  // correction for a residual the outer loop still considered unconverged on a stiff
  // tendon island (axial stiffness 1e5 N/m: the preconditioned norm is orders of magnitude
  // below the plain one) - a "Zero Preconditioner-dot product" PCG breakdown and a MINRES
  // fallback stalled at 2e-8 (2026-09-02).
  ColumnVector<real> rhsCopy(numDofs);
  AsView(rhsCopy) = rhs;
  real const outerThreshold = Max(
      backpropParams.outerSolverAbsTol, backpropParams.outerSolverRelTol * rhsCopy.Norm());
  innerLParams.normType = LinearSolverConvergenceNorm::ResidualL2;
  innerLParams.absTol = Min(backpropParams.innerSolverAbsTol, 0.1_r * outerThreshold);

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
    precLinearSolver.Solve(
        approxHessian, in, out, /*hasOperatorChanged*/ false, InitialGuessHint::Zero);
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
      backpropParams.verbosity,
      /*usePolakRibiere*/ true,
      InitialGuessHint::Zero);

  // If PCG diverged, fall back to MINRES.
  if (outerResult.convergence == LinearSolverConvergenceStatus::Diverged) {
    islandBackPropSolverStats.usedMinresFallback = true;
    if (backpropParams.verbosity >= VerbosityLevel::Warning) {
      MOCHI_LOG_WARNING(
          "PCG diverged at iteration %d (likely non-SPD). Falling back to MINRES.",
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
        backpropParams.verbosity,
        InitialGuessHint::Zero);

    if (outerResult.convergence == LinearSolverConvergenceStatus::Diverged &&
        backpropParams.verbosity >= VerbosityLevel::Warning) {
      MOCHI_LOG_WARNING("MINRES diverged at iteration %d.", outerResult.numIterDone);
    } else if (backpropParams.verbosity >= VerbosityLevel::Verbose) {
      MOCHI_LOG(
          "MINRES: Finished after %d iterations, final resNorm = %f, converged = %d",
          outerResult.numIterDone,
          outerResult.residualNorm,
          IsConverged(outerResult.convergence));
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
        IsConverged(outerResult.convergence));
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

  // Refinement against the true residual. The Krylov solve above stops on its recurrence
  // residual, which the finite-difference operator's noise lets drift from the true one, and it
  // works with the symmetric operator H while the adjoint operator carries the antisymmetric
  // chart term of external torques, J^T = H + 1/2 [tau]x (AddRigidTorqueChartTerm). Both are
  // handled by defect correction on the true residual: res = rhs - J^T z from fresh products,
  // solve H dz = res, z += dz. On an island with a chart term the rounds run to convergence:
  // while the residual is above the outer threshold and each round still lowers it (the
  // correction contracts by |H^-1 1/2 [tau]x| per round, and its fixed point is the solution
  // of the full operator; the former fixed budget of eight rounds is now a stall test and a
  // budget of 32). On an island without one the plain solve is the solution already, up to the
  // drift of the recurrence, and rounds are run only when the residual is above the acceptance
  // threshold defined below, to bring the solve within the contract; rounds on a solve that is
  // within it would only fit the noise of the products (a round at the noise floor lowers the
  // measured residual without bringing z closer to the solution, and moved gradients on stiff
  // islands by 1e-5 relative). A round that does not lower the residual (by half, in the rescue
  // case) is discarded and ends the refinement. Each round asks its solve for a hundredth of
  // the defect (or the outer threshold), not for the outer tolerance relative to the defect: a
  // defect at the noise floor cannot be reduced by 1e-10, and a solve chasing that accumulates
  // the products' noise over its iterations. The round budget is a safeguard, not the
  // definition of success: the verdict below is taken on the true residual of the returned z.
  auto const& descendants = reg.get<CIslandDescendants const>(island);
  real constexpr kRoundRelTol = 0.01_r;
  krylov::StatusResidualL2<krylov::UsualDot, real> roundStatusCheck(
      kRoundRelTol, outerThreshold, static_cast<real>(newtonParamsForward.relDivTol));
  bool hasChartTerm = false;
  auto trueResidual = [&](ColumnVectorView<real const> z, ColumnVectorView<real> outRes) {
    hessianOp(z, outRes);
    hasChartTerm = AddRigidTorqueChartTerm(reg, descendants, 1_r, z, outRes) || hasChartTerm;
    outRes *= -1_r;
    outRes += rhs; // rhs - J^T z
    return outRes.Norm();
  };
  // The round-off level of the operator bounds what any solve can reach: a central difference
  // of the residual with step eps carries round-off of the order of machine epsilon over eps
  // times the scale of what is differenced, so the residual rhs - J^T z cannot be driven below
  // that level times the scale of the products (|rhs| + |J^T z|), whatever the loss and the
  // scene. The level depends on the build's precision and on epsFiniteDiff (2e-8 in double at
  // the default step, 6e-4 in single), so no fixed tolerance expresses it. Measuring it instead
  // (the difference of two plain quotients at the step and at twice the step) was tried and
  // dropped: for a residual that is linear along z the rounding of the two quotients cancels
  // and the difference reads zero while the solve stalls at the round-off level all the same,
  // and in single precision a step below the configured one can fall under the state's
  // resolution. The analytic operator carries machine epsilon only.
  //
  // The contract: a solve is converged when its true residual is a finite number at or below
  // the acceptance threshold, which is the outer threshold or kFloorFactor times the round-off
  // level, whichever is larger. The factor covers what a solve reaches above the level: it
  // accumulates the products' round-off over its iterations, and the scale of what the residual
  // assembly differences exceeds the products' scale by the state's magnitude, the stiffness
  // of the island and the operator's conditioning (measured on the suite: up to 126 levels in
  // double precision on velocity gradients through rigid contact and 284 on a rod, up to 33
  // in single precision). A solve that failed sits orders of magnitude above that: an
  // iteration budget that ran out, a divergence, or a refinement that stalled above the
  // threshold all end reported as not converged. The level is a model, and a crude one in
  // both directions (a free body in the air has products that are exact to round-off far
  // below it, a rod's are noisier than it), so the refinement above does not use it to decide
  // when to stop: it uses its own measured progress, and runs while the residual is above the
  // outer threshold.
  real constexpr kFloorFactor = MOCHI_USE_DOUBLE_PRECISION ? 1024_r : 64_r;
  real const roundOff = std::numeric_limits<real>::epsilon() /
      (useAnalyticHvp ? 1_r : backpropParams.epsFiniteDiff);
  real const rhsNorm = rhs.Norm();
  MOCHI_FILO_STACK_ALLOCATOR(refineAllocator, 6 * 256 * sizeof(real));
  ColumnVector<real> res(rhs.Rows(), &refineAllocator);
  real resNorm = trueResidual(AsConstView(outZ), AsView(res));
  ColumnVector<real> jz(rhs.Rows(), &refineAllocator);
  auto roundOffLevel = [&](ColumnVectorView<real const> resNow) {
    jz = rhs;
    jz -= resNow; // J^T z
    return roundOff * (rhsNorm + jz.Norm());
  };
  auto acceptanceOf = [&](ColumnVectorView<real const> resNow) {
    return Max(outerThreshold, kFloorFactor * roundOffLevel(resNow));
  };
  {
    ColumnVector<real> zTrial(rhs.Rows(), &refineAllocator);
    ColumnVector<real> resTrial(rhs.Rows(), &refineAllocator);
    int constexpr kMaxRefinementRounds = 32;
    real constexpr kRescueGain = 0.5_r;
    for (int round = 0; round < kMaxRefinementRounds; ++round) {
      if (!std::isfinite(resNorm) || resNorm <= outerThreshold) {
        break;
      }
      if (!hasChartTerm && !(resNorm > acceptanceOf(AsConstView(res)))) {
        break; // a torque-free solve within the contract: the plain solve stands
      }
      zTrial.SetZero();
      auto const resView = AsConstView(res);
      auto dzView = AsView(zTrial);
      auto roundResult = krylov::PCG(
          hessianOp,
          resView,
          dzView,
          precOp,
          backpropParams.outerSolverMaxIter,
          roundStatusCheck,
          /*abortIfNotSpd*/ true,
          backpropParams.verbosity,
          /*usePolakRibiere*/ true,
          InitialGuessHint::Zero);
      if (roundResult.convergence == LinearSolverConvergenceStatus::Diverged) {
        zTrial.SetZero();
        krylov::StatusImplicitResidualNorm<real> roundMinresStatusCheck(
            kRoundRelTol, outerThreshold, static_cast<real>(newtonParamsForward.relDivTol));
        roundResult = krylov::MinRes(
            hessianOp,
            resView,
            dzView,
            precOp,
            backpropParams.outerSolverMaxIter,
            roundMinresStatusCheck,
            backpropParams.verbosity,
            InitialGuessHint::Zero);
        if (roundResult.convergence == LinearSolverConvergenceStatus::Diverged) {
          if (backpropParams.verbosity >= VerbosityLevel::Warning) {
            MOCHI_LOG_WARNING(
                "Adjoint refinement: the symmetric solve diverged in round %d.", round);
          }
          break;
        }
      }
      zTrial += outZ;
      real const trialNorm = trueResidual(AsConstView(zTrial), AsView(resTrial));
      real const gain = hasChartTerm ? 1_r : kRescueGain;
      if (!(trialNorm < gain * resNorm)) {
        break; // no progress (or a non-finite trial): keep z as it is
      }
      outZ = zTrial;
      res = resTrial;
      resNorm = trialNorm;
      outerResult.numIterDone += roundResult.numIterDone;
    }
  }

  // The true residual of the returned solution (the full operator J^T, from fresh products:
  // MINRES's implicit residual can be far from it, see the integrity check above) replaces the
  // solver's estimate, so that the reported adjoint residual never under-reports.
  outerResult.residualNorm = static_cast<double>(resNorm);

  real const residualFloor = roundOffLevel(AsConstView(res));
  real const acceptance = Max(outerThreshold, kFloorFactor * residualFloor);
  islandBackPropSolverStats.residualThreshold = static_cast<double>(acceptance);
  islandBackPropSolverStats.residualFloor = static_cast<double>(residualFloor);
  islandBackPropSolverStats.converged = std::isfinite(resNorm) && resNorm <= acceptance;

  // With validation on, also probe the symmetry of the operator. The adjoint solve assumes
  // H = H^T (the residual is the gradient of one merit function), which both PCG and MINRES rely
  // on; the probe compares rhs.(H z) with z.(H rhs), which agree for a symmetric H up to the
  // finite-difference noise of the products. Two more products per solve.
  // The probe is about H alone: with the chart term in one product, rhs.(J^T z) against
  // z.(H rhs) reported the chart term itself as an asymmetry (18 percent on a synthetic case).
  if (backpropParams.validateFiniteDiff) {
    ColumnVector<real> hz(rhs.Rows(), &refineAllocator);
    ColumnVector<real> hRhs(rhs.Rows(), &refineAllocator);
    hessianOp(AsConstView(outZ), AsView(hz));
    real const rhsDotHz = hz.Dot(rhs);
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
  islandBackPropSolverStats.usedMinresFallback = false;

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

      // Finite-difference approximation of dres * z, plus the chart term of J^T
      // (AddRigidTorqueChartTerm; the Newton iteration converges to the solution of the full
      // operator with the symmetric approximate dresidual as its Jacobian).
      ColumnVector<real> dresTimesZ(problem.GetDofsSize(), &allocator);
      GetHessianVectorProduct(
          reg, island, GradTarget::Current, problemForward, problem.GetSolution(), dresTimesZ);
      AddRigidTorqueChartTerm(
          reg,
          reg.get<CIslandDescendants const>(island),
          1_r,
          AsConstView(problem.GetSolution()),
          AsView(dresTimesZ));

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

  // If some actor has contact queries enabled, run collision detection over the full island
  // (the queried actors may act as colliders) and one assembly at the prepared state: the
  // assembly stores the forces of the queried pairs in the contact containers, from which
  // PrepareBackPropagation refreshes the queries. The engine's queries are outputs of a step,
  // not state: after a state restore they still report the last forward step, so a loss on a
  // restored step (a running contact-force loss) would read the wrong force. The assembly's
  // forces equal the forward step's (same state, same detection). The dresidual part
  // initializes the fresh problem's per-actor storage, which a residual assembly needs.
  if (std::any_of(descendants.actors.begin(), descendants.actors.end(), [&](auto const& e) {
        return reg.any_of<TagQueryActiveContacts>(e);
      })) {
    GetSolutions(problem.solution, reg, descendants.actors, /*baseOffset*/ 0);
    AssemblyParams const params = {
        .assemObj = false, .assemRes = true, .assemDRes = true, .psdDRes = true};
    solver::AssembleIslandPipeline(reg, island, params, problem);
  }
}

void mochi::PrepareBackPropagation(entt::registry& reg) {
  MOCHI_PROFILE_SCOPE();

  // The contact detection of the prepared state computes the current SDF Hessians for the
  // contact-force adjoints (see TagAdjointContactDetection).
  reg.set<TagAdjointContactDetection>();
  TaskSemaphore eachTask;
  reg.view<CIslandDescendants const>().each(
      [&](entt::entity island, CIslandDescendants const& descendants) {
        Schedule(eachTask, "PrepareBackPropagationIslandAsync", [&, island]() {
          PrepareBackPropagationIslandAsync(reg, island, descendants);
        });
      });
  eachTask.Wait();
  reg.unset<TagAdjointContactDetection>();

  // The queries (contact forces, contact points, ...) of the prepared state, from the forces
  // the islands' assemblies stored (see PrepareBackPropagationIslandAsync).
  UpdateAllActorQueries(reg);

  ecs::InvokeForEachGlobal(&PrepareContactForceAdjoints, reg);

  // The contact-parameter gradient components of the islands' contact owners, created here,
  // sequentially: the island solves (running in parallel) accumulate the direct term of the
  // contact-force adjoints into them, and the parameter stage the state path.
  reg.view<CIslandDescendants const>().each([&](CIslandDescendants const& descendants) {
    for (auto const e : descendants.actors) {
      if (reg.try_get<CContactParams const>(e) != nullptr &&
          reg.try_get<CDiffContactParamsGrad>(e) == nullptr) {
        reg.emplace<CDiffContactParamsGrad>(e);
      }
    }
  });
}

static void AccumulateContactForceParameterGradientsIsland(
    entt::registry& reg,
    entt::entity island,
    CIslandDescendants const& descendants,
    SnleProblem<real>& problemForward);

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

  // 5. The direct term of the contact-force adjoints in the contact parameters.
  AccumulateContactForceParameterGradientsIsland(reg, island, descendants, problemForward);
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
// Visits every contact-detection result of the island that carries contact-force adjoints
// (the collisions of the queried actors, and the collisions of other actors against a queried
// collider), each once, with the pair's identity (colliding actor, collider).
template <typename Fn>
static void ForEachQueriedCollisionResult(
    entt::registry& reg,
    CIslandDescendants const& descendants,
    Fn&& fn) {
  for (auto const e : descendants.actors) {
    if (!reg.all_of<TagQueryActiveContacts>(e)) {
      continue;
    }
    if (auto* collisions =
            reg.try_get<CActiveCollisions<ContactType::Async, TimeStep::Current>>(e)) {
      for (auto& collision : *collisions) {
        fn(e, collision.colliderEntity, collision.collisionResult);
      }
    }
    if (auto* collisions =
            reg.try_get<CActiveCollisions<ContactType::Sync, TimeStep::Current>>(e)) {
      for (auto& collision : *collisions) {
        fn(e, collision.colliderEntity, collision.collisionResult);
      }
    }
    if (auto* colliderJacs = reg.try_get<CCollJacs<CollRole::Collider>>(e)) {
      for (auto& jac : *colliderJacs) {
        // The results of a queried colliding actor are visited through its own collisions.
        if (!reg.all_of<TagQueryActiveContacts>(jac.otherEntity)) {
          fn(jac.otherEntity, e, *jac.query);
        }
      }
    }
  }
}

// The per-contact force adjoints of one contact pair, copied out of the contact containers:
// every assembly at a changed solution reruns collision detection, which rebuilds the
// containers (the adjoint solve's finite-difference products do), so the direct parameter
// term below matches the adjoints to the rebuilt contacts by sample index.
struct ContactForceAdjointStashEntry {
  entt::entity colliding = entt::null;
  entt::entity collider = entt::null;
  std::vector<int> sampleIndices;
  std::vector<Real3> adjoints;
};

// Registry context: the stash of every island with contact-force adjoints, built by
// BackPropagationSolve before the island solves and read by them.
struct CtxContactForceAdjointStash {
  std::unordered_map<entt::entity, std::vector<ContactForceAdjointStashEntry>> byIsland;
};

static void StashContactForceAdjoints(entt::registry& reg) {
  auto& stash = reg.ctx_or_set<CtxContactForceAdjointStash>();
  stash.byIsland.clear();
  reg.view<CIslandDescendants const>().each(
      [&](entt::entity island, CIslandDescendants const& descendants) {
        std::vector<ContactForceAdjointStashEntry> entries;
        ForEachQueriedCollisionResult(
            reg,
            descendants,
            [&](entt::entity colliding, entt::entity collider, ContactDetectionResult const& result) {
              bool any = false;
              for (auto const& adjoint : result.forceAdjoint) {
                any = any || adjoint[0] != 0_r || adjoint[1] != 0_r || adjoint[2] != 0_r;
              }
              if (!any) {
                return;
              }
              MOCHI_ASSERT(
                  isize(result.forceAdjoint) == isize(result.sampleIndices),
                  "Unexpected number of contact-force adjoints.");
              ContactForceAdjointStashEntry entry;
              entry.colliding = colliding;
              entry.collider = collider;
              entry.sampleIndices.assign(
                  result.sampleIndices.begin(), result.sampleIndices.end());
              entry.adjoints.assign(result.forceAdjoint.begin(), result.forceAdjoint.end());
              entries.push_back(std::move(entry));
            });
        if (!entries.empty()) {
          stash.byIsland[island] = std::move(entries);
        }
      });
}

// The contact-detection result of the pair (colliding, collider) in the current containers.
static ContactDetectionResult const* FindCollisionResult(
    entt::registry& reg,
    entt::entity colliding,
    entt::entity collider) {
  if (auto* collisions =
          reg.try_get<CActiveCollisions<ContactType::Async, TimeStep::Current>>(colliding)) {
    for (auto& collision : *collisions) {
      if (collision.colliderEntity == collider) {
        return &collision.collisionResult;
      }
    }
  }
  if (auto* collisions =
          reg.try_get<CActiveCollisions<ContactType::Sync, TimeStep::Current>>(colliding)) {
    for (auto& collision : *collisions) {
      if (collision.colliderEntity == collider) {
        return &collision.collisionResult;
      }
    }
  }
  return nullptr;
}

// Sum_s lambda_s . f_s over the stashed contacts of the island: the contact-force adjoints
// against the forces the last residual assembly stored (forcePerUnitArea), matched by sample
// index (a sample at the margin of the detection tolerance may enter or leave with a perturbed
// penalty coefficient; its force is zero there).
static real DotContactForceAdjoints(
    entt::registry& reg,
    std::vector<ContactForceAdjointStashEntry> const& entries) {
  real sum = 0_r;
  std::unordered_map<int, int> positions;
  for (auto const& entry : entries) {
    ContactDetectionResult const* result = FindCollisionResult(reg, entry.colliding, entry.collider);
    MOCHI_ASSERT(result != nullptr, "A queried contact pair disappeared from the containers.");
    MOCHI_ASSERT(
        isize(result->forcePerUnitArea) == isize(result->sampleIndices),
        "The residual assembly did not store the forces of a queried contact pair.");
    positions.clear();
    for (int j = 0; j < isize(result->sampleIndices); ++j) {
      positions[result->sampleIndices[j]] = j;
    }
    for (int i = 0; i < isize(entry.sampleIndices); ++i) {
      auto const found = positions.find(entry.sampleIndices[i]);
      if (found == positions.end()) {
        continue;
      }
      sum += Get0(VDot<3>(
          ToSimd(entry.adjoints[i]), ToSimd(result->forcePerUnitArea[found->second])));
    }
  }
  return sum;
}

// The finite-difference step of a contact parameter. A contact pair combines both owners'
// values by geometric mean (CombineContactParams); the dissipative coefficients (Coulomb,
// viscous, normal damping) must be non-negative and the penalty coefficient is strictly
// positive. A positive coefficient c is perturbed multiplicatively, c e^{+-eps}, which never
// leaves the admissible domain however small c is, and the central difference quotient in
// log c is divided by c: the pair's coefficient Sqrt(c c') is smooth in log c, so the estimate
// is second-order accurate at any positive value (an additive step of the order of c evaluated
// the negative side of a small c - 41% off at c = 1e-8 - and for c of order one the two steps
// coincide). At a zero-valued owner coefficient the loss is a square-root cusp in that
// coefficient when the partner's is positive (derivative +inf) and flat when it is zero; the
// right-sided difference quotient at the finite-difference step is used there instead: it is
// exactly zero for a zero partner and grows like 1 / Sqrt(step) otherwise - a finite, correctly
// signed push for optimizers constrained to non-negative coefficients (test_diffsim_params.py
// pins both behaviors).
struct ContactParamPerturbation {
  real saved;
  real plus;
  real minus;
  real scale; // the difference quotient's factor: d/dc = scale * (f(plus) - f(minus))
  ContactParamPerturbation(real value, real epsFiniteDiff) : saved(value) {
    if (value > 0_r) {
      plus = value * std::exp(epsFiniteDiff);
      minus = value * std::exp(-epsFiniteDiff);
      scale = 1_r / (2_r * epsFiniteDiff * value);
    } else {
      real const eps = epsFiniteDiff * (1_r - value);
      plus = value + eps;
      minus = value;
      scale = 1_r / eps;
    }
  }
};

static void ContactParamFields(
    CContactParams& contactParams,
    real* (&outFields)[kNumContactParamGradients]) {
  outFields[0] = &contactParams.penaltyCoefficient;
  outFields[1] = &contactParams.coulombFrictionCoefficient;
  outFields[2] = &contactParams.viscousFrictionCoefficient;
  outFields[3] = &contactParams.normalViscousDampingCoefficient;
}

// The contact-force adjoints reach the contact parameters directly, at fixed states: a query
// F = Sum_s w_s J_s^T f_s(p_s; theta) on an actor of the island depends on the pair parameters
// besides the state path the adjoint solve covers. Every residual assembly stores the forces of
// the queried pairs (forcePerUnitArea), so central differences of the assembly in each owner's
// parameters give Sum_s lambda_s . df_s/dtheta, with lambda_s the per-contact force adjoints
// (forceAdjoint, stashed by BackPropagationSolve before the solves rebuild the containers).
// Evaluated right after the island's solve, at the exact step states. (Until 2026-09-07 this
// term was missing: the Coulomb gradient of a sliding cube's contact-force loss was 85% too
// small.)
static void AccumulateContactForceParameterGradientsIsland(
    entt::registry& reg,
    entt::entity island,
    CIslandDescendants const& descendants,
    SnleProblem<real>& problemForward) {
  MOCHI_PROFILE_SCOPE();
  auto const* stash = reg.try_ctx<CtxContactForceAdjointStash const>();
  if (stash == nullptr) {
    return;
  }
  auto const found = stash->byIsland.find(island);
  if (found == stash->byIsland.end()) {
    return;
  }
  std::vector<ContactForceAdjointStashEntry> const& entries = found->second;
  auto const& solverParams = reg.ctx<CBackPropagationSolverParams const>();
  AssemblyParams params = {.assemObj = false, .assemRes = true, .assemDRes = false};
  problemForward.SetAssemblyFunction(
      [&](SnleProblem<real>& problem, AssemblyParams const& /* params */) {
        solver::AssembleIslandPipeline(reg, island, params, problem);
      });
  MOCHI_FILO_STACK_ALLOCATOR(allocator, 256 * sizeof(real));
  ColumnVector<real> delta(problemForward.GetDofsSize(), &allocator);
  delta.SetZero();
  // The zero increment re-anchors the actors at the exact solution (the products of the solve
  // left them at their last perturbed evaluation point); the parameter perturbations below
  // never move the state.
  auto evalForces = [&]() {
    solver::PostNewIncrementLocalPipeline(reg, island, delta, problemForward.solution);
    problemForward.InvalidateCachedData();
    problemForward.UpdateResidual();
    return DotContactForceAdjoints(reg, entries);
  };
  for (auto const e : descendants.actors) {
    auto* contactParams = reg.try_get<CContactParams>(e);
    if (contactParams == nullptr) {
      continue;
    }
    auto& outContactGrad = reg.get<CDiffContactParamsGrad>(e); // created by PrepareBackPropagation
    real* fields[kNumContactParamGradients];
    ContactParamFields(*contactParams, fields);
    for (int f = 0; f < kNumContactParamGradients; ++f) {
      ContactParamPerturbation const step(*fields[f], solverParams.epsFiniteDiff);
      *fields[f] = step.plus;
      real const forcesPlus = evalForces();
      *fields[f] = step.minus;
      real const forcesMinus = evalForces();
      *fields[f] = step.saved;
      outContactGrad.value[f] += step.scale * (forcesPlus - forcesMinus);
    }
  }
  // Leave the actors at the exact solution and the containers with the unperturbed forces.
  evalForces();
}

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

  // Contact parameters, per owner (standalone actors and nested links alike): the state path
  // -lambda^T dR/dtheta, with the perturbation rule of ContactParamPerturbation. The direct
  // dependence of contact-force queries on the parameters is accumulated by
  // AccumulateContactForceParameterGradientsIsland, right after the adjoint solve.
  for (auto const e : descendants.actors) {
    auto* contactParams = reg.try_get<CContactParams>(e);
    if (contactParams == nullptr) {
      continue;
    }
    auto& outContactGrad = reg.get<CDiffContactParamsGrad>(e); // created above
    real* fields[kNumContactParamGradients];
    ContactParamFields(*contactParams, fields);
    for (int f = 0; f < kNumContactParamGradients; ++f) {
      ContactParamPerturbation const step(*fields[f], solverParams.epsFiniteDiff);
      *fields[f] = step.plus;
      evalResidual(AsView(residualPlus));
      *fields[f] = step.minus;
      evalResidual(AsView(residualMinus));
      *fields[f] = step.saved;
      residualPlus -= residualMinus;
      outContactGrad.value[f] += -step.scale * lambda.Dot(residualPlus);
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
  // zero in the assembly (a non-positive coefficient disables the term), so a central
  // difference whose lower sample would cross zero - at massDampingCoefficient == 0, and for
  // any positive coefficient below the step - would straddle the gate and report about half
  // the derivative (49.95 percent at 1e-6, 45 percent at 1e-4, measured on the extracted
  // stencil); the right-sided difference is used whenever the lower sample would leave the
  // domain - the direction an optimizer constrained to alpha >= 0 can move in - and the same
  // rule keeps Young's modulus and the density positive. The three terms are affine in their
  // coefficient, so a one-sided difference within the domain is as exact as a central one.
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
      bool const rightSided = value - h < 0_r; // the lower sample would leave the domain
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

  // The contact-force adjoints leave the contact containers with the first assembly below.
  StashContactForceAdjoints(reg);

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

  // The island pre-step operation and the time-integrator state of the restored step (as in
  // PrepareBackPropagationIslandAsync): without them the assemblies would use the stage size of
  // whichever step ran last, wrong as soon as consecutive steps differ in size.
  PreStepIslandAsync(reg, descendants);
  auto const& simParams = reg.ctx<CSimulationParams const>();
  MOCHI_ASSERT(
      simParams.integrationMethod == IntegrationMethod::BackwardEuler,
      "Only Backward Euler is supported")
  auto const integrationParams =
      solver::CreateIslandTimeIntegrationParams(reg, descendants, simParams.integrationMethod);
  MOCHI_ASSERT(integrationParams.numStages == 1);
  solver::SetTimeIntegratorState(reg, descendants.actors, integrationParams, /*iStage*/ 0);

  // Set the stage-start state
  solver::PreFirstStageLocalPipeline(reg, descendants);
  solver::PreStageLocalPipeline(reg, descendants, problem);

  // Assemble problem with psdDRes=false, notifying exact Hessian
  AssemblyParams paramsExactDRes = {
      .assemObj = false, .assemRes = false, .assemDRes = true, .psdDRes = false};
  problem.UpdateObjResDRes(paramsExactDRes);

  // Direct factorization using LU of the step Jacobian: the fixed-chart Hessian plus the
  // moving-chart term of the external torques (AddRigidTorqueChartTerm).
  Matrix<real> dResCurr = ToMatrix(problem.GetDResidual());
  AddRigidTorqueChartTerm(reg, descendants, -1_r, dResCurr);
  LU<real> invDRes(dResCurr);

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
  outGradDerivedStep.stepDt = 0.0;
  if (outTargetPoseGrad) {
    outTargetPoseGrad->propagated.SetZero();
  }
}

void mochi::PrepareContactForceAdjoints(
    ecs::RequiredTag<TagQueryActiveContacts>,
    [[maybe_unused]] CRequiresFarSdfEvaluation const* farSdfEval,
    CActiveCollisions<ContactType::Async, TimeStep::Current>& outActiveCollisionsAsync,
    CActiveCollisions<ContactType::Sync, TimeStep::Current>& outActiveCollisionsSync,
    CCollJacs<CollRole::Collider>* outColliderJacs) {
  MOCHI_ASSERT_VERBOSE(
      !farSdfEval, "Far SDF evaluation is not compatible with contact-force queries.");

  auto prepareAdjoints = [](ContactDetectionResult& collisionResult) {
    auto& adjoints = collisionResult.forceAdjoint;
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
