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

#include "mochi_contact.h"
#include "mochi_ecs_utils.h"
#include "mochi_island.h"
#include "mochi_simulation.h"
#include "mochi_snle.h"

#include <array>

namespace mochi::diffsim {
// Forward declarations: this header only aliases these types as ECS components.
struct BackPropagationSolverParams;
struct BackPropagationSceneStats;
} // namespace mochi::diffsim

namespace mochi {

struct CSoftMaterialParams;

/**************************************************************************
  ECS components for differentiable scenes, islands and actors
*/

// Tag to denote a differentiable scene.
struct TagDifferentiableScene {};

// Tag set by PrepareBackPropagate and cleared by BackPropagate.
struct TagBackPropagationPrepared {};

// Set while PrepareBackPropagation runs the contact detection of the prepared state: that
// detection also computes the current-state SDF Hessians, which the contact-force adjoints
// (AccumulateContactForceAdjoints) need to differentiate the penalty force through the SDF
// gradient. The forward solve never computes them.
struct TagAdjointContactDetection {};

// Stores the state pair restored by PrepareBackPropagate.
struct CStatePair {
  StateHandle stateNew;
  StateHandle stateOld;
  // Time step of the prepared step (stateOld -> stateNew), recorded by
  // SceneImpl::PrepareBackPropagate. The velocity backward accessors need it after the sweep,
  // when the scene holds the restored old state whose own delta time is that of the step
  // before it (the default 1e-2 for the initial state).
  double stepDt = 0.0;
};

// Tag to denote a constraint with differentiable input (e.g. pose controller).
struct TagConstraintWithDifferentiableInput {};

// Component to store the differentiability solver parameters.
using CBackPropagationSolverParams = diffsim::BackPropagationSolverParams;

// Component to store the performance metrics of the last back-propagation step.
using CBackPropagationSceneStats = diffsim::BackPropagationSceneStats;

// Stores information on the number of DoFs of the derived state of an island, necessary for sizing
// derived state vectors.
struct CIslandDerivedStateInfo : NoCopy {
  int dofsSize = 0;
};

// Stores information on the number of DoFs of the derived state of an actor, necessary for indexing
// derived state vectors.
struct CActorDerivedStateInfo : NoCopy {
  int dofsSize = 0;
};

// Stores the DoF offset of the derived state of an actor within an island, necessary for indexing
// derived state vectors.
struct CDerivedStateOffset : NoCopy {
  int dofsOffset = 0;
};

// Stores information on the size of the differentiable input of an island, necessary for sizing
// input vectors.
struct CIslandDiffInputInfo : NoCopy {
  int size = 0;
};

// Stores information on the statistics of an island backprop solve execution.
struct CIslandBackPropSolverStats : NoCopy {
  StageSolverStats stats;
  // True if every finite-difference Hvp this back-prop evaluated (across the whole
  // outer solve) converged: its quotient agreed with the one at half the step size
  // to 1e-2, after at most four halvings (GetHessianVectorProduct). Only meaningful
  // when BackPropagationSolverParams::validateFiniteDiff is set; otherwise stays at
  // its default of true. Aggregated into BackPropagationSceneStats::finiteDiffValid
  // via logical AND across islands.
  bool finiteDiffValid = true;
  // Relative asymmetry of the adjoint operator measured by KrylovSolveZ's symmetry probe
  // (only with validateFiniteDiff; 0 otherwise). Aggregated into
  // BackPropagationSceneStats::hessianAsymmetry via max across islands.
  double hessianAsymmetry = 0.0;
  // True if the PCG adjoint solve of this island aborted (non-SPD detection or a
  // preconditioner breakdown) and the solution comes from the MINRES fallback. Counted into
  // BackPropagationSceneStats::numMinresFallbacks.
  bool usedMinresFallback = false;
  // Whether the island's adjoint solve met its acceptance threshold (a finite true residual at
  // or below the larger of the outer threshold and 1024 times (64 in single precision) the operator's round-off level),
  // that threshold, and the round-off level. Aggregated into BackPropagationSceneStats (see
  // ComputeAggregateBackPropSolverSceneStats).
  bool converged = true;
  double residualThreshold = 0.0;
  double residualFloor = 0.0;
};

// Stores information on the size of the differentiable input of an actor, necessary for indexing
// input vectors.
struct CActorDiffInputInfo : NoCopy {
  int dofsSize = 0;
};

// Stores the offset of the differentiable input of an actor within an island, necessary for
// indexing input vectors.
struct CDiffInputOffset : NoCopy {
  int dofsOffset = 0;
};

// Store the DoF offset of the actor in the scene
struct CSceneStateOffset : NoCopy {
  int dofsOffset = 0;
};

// Components to store adjoints. CDiffDerivedStepGrad is only used internally across calls to
// BackPropagate. CDiffStateGrad OTOH receives external gradients in GetFooBackward() functions, and
// exposes gradients for SetFooBackward() functions.
struct CDiffStateGrad {
  ColumnVector<real> value;

  CDiffStateGrad() = default;
  explicit CDiffStateGrad(int size) : value(size) {}

  MOCHI_STRUCT_BEGIN(mochi::CDiffStateGrad);
  MOCHI_ATTRIBUTE(CaptureState);
  MOCHI_ATTRIBUTE(HasAdjoint);
  MOCHI_FIELD(value);
  MOCHI_STRUCT_END();
};
struct CDiffDerivedStepGrad {
  // Adjoint of the previous step's delta, as assembled by the back-propagation of the step
  // that consumed it (GradTarget::PreviousDelta), i.e. under that step's model of the previous
  // velocity v_{k-1} = delta_{k-1} / dt_k.
  ColumnVector<real> value;
  // Time step of the back-propagated step that assembled `value` (0 after a reset). The delta
  // itself was produced by the previous step, whose own dt_{k-1} defines v_{k-1} = delta / dt_{k-1};
  // folding `value` into that step therefore scales it by dt_k / dt_{k-1} (SceneImpl::BackPropagate),
  // which is the identity for uniform step sizes only.
  double stepDt = 0.0;

  CDiffDerivedStepGrad() = default;
  explicit CDiffDerivedStepGrad(int size) : value(size) {}

  MOCHI_STRUCT_BEGIN(mochi::CDiffDerivedStepGrad);
  MOCHI_ATTRIBUTE(CaptureState);
  MOCHI_ATTRIBUTE(HasAdjoint);
  MOCHI_FIELD(value);
  MOCHI_FIELD(stepDt);
  MOCHI_STRUCT_END();
};
// Component to store contact-force adjoints, templatized by GradTarget.
template <GradTarget kGradTarget>
struct CDiffContactGrad : public ColumnVector<real> {
  using ColumnVector<real>::ColumnVector;
};
// Components to store temporary gradient data during the back-propagation step.
struct CDiffContainerState : public ColumnVector<real> {
  using ColumnVector<real>::ColumnVector;
};
struct CDiffContainerDerivedState : public ColumnVector<real> {
  using ColumnVector<real>::ColumnVector;
};
// Component to store gradients wrt pose targets computed during the back-propagation step.
struct CDiffTargetPoseGrad : NoCopy {
  ColumnVector<real> current; // Gradient wrt the current target pose
  ColumnVector<real> previous; // Gradient wrt the previous target pose
  ColumnVector<real> propagated; // Gradient accumulated from previous steps

  CDiffTargetPoseGrad() = default;
  explicit CDiffTargetPoseGrad(int size)
      : current(ColumnVector<real>::Zero(size)),
        previous(ColumnVector<real>::Zero(size)),
        propagated(ColumnVector<real>::Zero(size)) {}

  MOCHI_STRUCT_BEGIN(mochi::CDiffTargetPoseGrad);
  MOCHI_ATTRIBUTE(CaptureState);
  MOCHI_ATTRIBUTE(HasAdjoint);
  MOCHI_FIELD(propagated);
  // `current` and `previous` are not captured.
  MOCHI_STRUCT_END();
};
// Component to store the gradient wrt external forces computed during the back-propagation step.
struct CDiffForceGrad : public ColumnVector<real> {
  using ColumnVector<real>::ColumnVector;
};
// Global context component accumulating the loss gradient with respect to the scene
// gravity vector across BackPropagate calls. Created/zeroed by ResetBackPropagation.
struct CDiffGravityGrad {
  Real3 value{};
};

// Number of per-owner contact-parameter gradients accumulated by the parameter
// adjoint, and their fixed order:
//   0 penaltyCoefficient, 1 coulombFrictionCoefficient, 2 viscousFrictionCoefficient,
//   3 normalViscousDampingCoefficient.
// frictionFalloffVel is deliberately excluded: it is consumed through data precomputed
// during contact preparation, so re-assembling the residual under a perturbed value
// does not observe it (measured: residual-FD gradient identically zero while the
// rollout gradient is nonzero). A correct falloff gradient requires re-running contact
// preparation per perturbation - future work, not a silent zero.
inline constexpr int kNumContactParamGradients = 4;

// Per-entity component accumulating dL/d(contact params) across BackPropagate calls,
// for every contact-parameter owner (standalone actors and nested links) of a
// back-propagated island. Created zeroed during the sweep so that "accumulated, zero"
// is distinguishable from "never part of a back-propagated island"; zeroed by
// ResetBackPropagation.
struct CDiffContactParamsGrad {
  std::array<real, kNumContactParamGradients> value{};
};

// Per-entity component accumulating dL/d(density) across BackPropagate calls, for
// every rigid-body-inertia owner (standalone rigid actors and articulated links) of a
// back-propagated island. Same existence semantics as CDiffContactParamsGrad.
struct CDiffDensityGrad {
  real value = 0_r;
};

// Number of per-actor soft material parameter gradients accumulated by the parameter
// adjoint, and their fixed order:
//   0 youngsModulus, 1 poissonRatio, 2 density, 3 massDampingCoefficient.
// Covers standalone soft actors with a homogeneous Lame-type material (Neo-Hookean,
// St. Venant-Kirchhoff, linear elastic). Other material models and per-element material
// fields are not accumulated; the readout reports them as errors. Mass damping is gated
// at zero in the assembly (a non-positive coefficient disables the term), so at
// massDampingCoefficient == 0 the accumulated value is the right-sided derivative.
inline constexpr int kNumSoftMaterialParamGradients = 4;
inline constexpr int kSoftMaterialGradYoungsModulus = 0;
inline constexpr int kSoftMaterialGradPoissonRatio = 1;
inline constexpr int kSoftMaterialGradDensity = 2;
inline constexpr int kSoftMaterialGradMassDamping = 3;

// Per-entity component accumulating dL/d(soft material parameters) across BackPropagate
// calls, for every supported soft actor of a back-propagated island. Same existence
// semantics as CDiffContactParamsGrad.
struct CDiffSoftMaterialGrad {
  std::array<real, kNumSoftMaterialParamGradients> value{};
};

// True when the soft material parameter adjoint covers this material: a homogeneous
// (single per-element entry) Lame-type parameter set.
bool IsSoftMaterialGradientSupported(CSoftMaterialParams const& material);


struct CForwardPropContainerDerivedStateJac {
  Matrix<real> data;
  // The following two fields are used to store the island this actor belongs to,
  // including other actors in the island as well as the total number of degrees of freedom.
  //
  // We store these information so that it is kept when we restore a new state, which overrides the
  // island information. Note that GetStepJacobian correlates three states: q_k,q_k-1,q_k-2.
  // In order to compute dq_k/dq_k-1, we need to restore to state q_k, where we record the island
  // information. Next, in order to compute dq_k/dq_k-2, we need to restore to state q_k-1, but we
  // need to use island information at q_k, which is stored here.
  DynamicArray<entt::entity> actors;
  int numIslandDofs = 0;
};

/**************************************************************************
  Utility functions
*/

// Function to obtain TimeStep from GradTarget. Should not be called with GradTarget::PreviousDelta.
template <GradTarget kGradTarget>
TimeStep constexpr GetTimeStep() {
  if constexpr (kGradTarget == GradTarget::Current || kGradTarget == GradTarget::CurrentInput) {
    return TimeStep::Current;
  } else {
    static_assert(
        kGradTarget == GradTarget::Previous || kGradTarget == GradTarget::PreviousInput,
        "Unexpected grad target");
    return TimeStep::StageStart;
  }
}

// Get the colliding position at the appropriate time step. Only supported for TimeStep::Current and
// TimeStep::StageStart.
template <TimeStep kTimeStep>
Real3 const& GetCollidingPosition(ContactDetectionResult const& data, int contact) {
  static_assert(
      kTimeStep == TimeStep::Current || kTimeStep == TimeStep::StageStart,
      "Only TimeStep::Current and TimeStep::StageStart store colliding positions");
  return (kTimeStep == TimeStep::Current) ? data.posColliding[contact]
                                          : data.posCollidingStageStart[contact];
}

/**************************************************************************
  Fore/Back propagation operations
*/

void PrepareBackPropagation(entt::registry& reg);

void BackPropagationSolve(entt::registry& reg);

// Accumulate every island's contribution to the parameter gradients (gravity into
// CDiffGravityGrad, contact parameters into per-owner CDiffContactParamsGrad,
// density into per-owner CDiffDensityGrad). Called by SceneImpl::BackPropagate after
// the island adjoint solves AND after re-restoring the exact step-state pair: the
// finite-difference Hessian-vector products leave the actors at their last perturbed
// evaluation point, and evaluating d(residual)/d(parameter) at that O(eps)-drifted
// state contaminates mass-proportional parameters (measured density-gradient error
// of 0.575 * eps/dt).
void AccumulateParameterGradients(entt::registry& reg);

// Zeroes one entity's accumulated contact-parameter gradient (ResetBackPropagation).
void ResetContactParamsGradContainers(CDiffContactParamsGrad& outGrad);

// Zeroes one entity's accumulated density gradient (ResetBackPropagation).
void ResetDensityGradContainers(CDiffDensityGrad& outGrad);

// Zeroes one entity's accumulated soft material gradient (ResetBackPropagation).
void ResetSoftMaterialGradContainers(CDiffSoftMaterialGrad& outGrad);

void ComputeHqx(
    int numIslandDofs,
    Span<entt::entity const> actors,
    CActorSnle const& actorSnle,
    CDofOffset const& dofOffset,
    CActorDerivedStateInfo const& derivedStateInfo,
    CForwardPropContainerDerivedStateJac& outDerivedState);

void ComputeDqDDerivedState(
    int numIslandDofs,
    LU<real> const& invDRes,
    Span<entt::entity const> actors,
    CActorSnle const& actorSnle,
    CDofOffset const& dofOffset,
    CActorDerivedStateInfo const& derivedStateInfo,
    CForwardPropContainerDerivedStateJac& outDerivedState);

void StepJacobianSolve(entt::registry& reg, MatrixView<real> jacCurr);

void StepJacobianShiftAndProject(
    entt::registry& reg,
    MatrixView<real> jacCurr,
    MatrixView<real> jacOld);

/**************************************************************************
  Per-actor systems
*/

// System to emplace differentiability components on actors
void EmplaceDifferentiabilityComponents(
    int numDerivedStateDofs,
    entt::registry& reg,
    entt::entity e,
    CActorDofInfo const& dofInfo);

// System to emplace components for differentiable contact forces
void EmplaceDifferentiableContactComponents(
    entt::registry& reg,
    entt::entity e,
    CActorDofInfo const& dofInfo);

// System to emplace differentiability components on constraints
void EmplaceConstraintDifferentiabilityComponents(entt::registry& reg, entt::entity e);

// System to reset back-propagation components between runs.
void ResetBackPropagationContainers(
    CDiffStateGrad& outGradState,
    CDiffDerivedStepGrad& outGradDerivedStep,
    CDiffTargetPoseGrad* outTargetPoseGrad);

// System to reset contact force adjoints before running backward contact queries. The forward
// assembly stores the contact forces in `ContactDetectionResult::forcePerUnitArea` for every
// pair with a contact query (TagQueryActiveContacts: contact points, node contact forces or
// the total contact force), and back-propagation uses that container for the force adjoints:
// it must start from zero for every such actor, not only for those with a total-force query.
void PrepareContactForceAdjoints(
    ecs::RequiredTag<TagQueryActiveContacts>,
    CRequiresFarSdfEvaluation const* farSdfEval,
    CActiveCollisions<ContactType::Async, TimeStep::Current>& outActiveCollisionsAsync,
    CActiveCollisions<ContactType::Sync, TimeStep::Current>& outActiveCollisionsSync,
    CCollJacs<CollRole::Collider>* outColliderJacs);

// System to accumulate contact force adjoints to actor level.
void AccumulateContactForceAdjoints(entt::registry& reg);

namespace differentiable {
void InitializeOnce(entt::registry& reg);
}

} // namespace mochi
