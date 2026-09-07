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

#include "mochi_articulated_body.h"
#include "mochi_contact.h"
#include "mochi_discretization_components.h"
#include "mochi_rigid.h"
#include "mochi_rod.h"

#include <mochi_core/contact/dmap.h>
#include <mochi_core/geometry/tetrahedral_map.h>
#include <mochi_core/utils/array_utils.h>

#include <mochi_core/memory/filo_allocator.h>

using namespace mochi;

static ContactEvalConfig InitializeContactForceAdjointEvalConfig(
    CSimulationParams const& simParams) {
  // Match the contact force law used by the prepared query data. Do not apply solver-stabilization
  // approximations here: PSD projection and fitted Hessians are useful for nonlinear solves, but
  // this VJP needs the derivative of the actual force returned by the query.
  return ContactEvalConfig{
      .psdDRes = false,
      .explicitNormals = simParams.experimentalEval.explicitNormals,
      .fadeFriction = simParams.experimentalEval.fadeFriction,
      .implicitNormalForceForDissipation =
          simParams.experimentalEval.implicitNormalForceForDissipation,
      .useFittedHessian = false,
      .frictionModel = simParams.experimentalEval.frictionModel};
}

static ContactEvalConfig RefineContactForceAdjointEvalConfig(
    ContactAssemblyReg const& reg,
    entt::entity colliding,
    ContactEvalConfig const& base) {
  ContactEvalConfig config = base;
  config.addPadding = ShouldAddPenaltyPadding(reg.get<CColliderInfo const>(colliding).type);
  config.validCollidingNormals = ValidCollidingNormals(reg, colliding);
  return config;
}

template <GradTarget kGradTarget>
static void AccumulateAsyncContactForceAdjoints(
    ContactAssemblyReg reg,
    entt::entity e,
    ecs::Included<TagRigidActor>,
    ecs::Excluded<TagStaticActor>,
    ecs::CtxGlobal<CSimulationParams const> simParams,
    CTimeIntegratorState const& intState,
    CRigidState<GetTimeStep<kGradTarget>()> const& pose,
    CQueryActorContactForces const& /*queryActorContactForces*/,
    CActiveCollisions<ContactType::Async, TimeStep::Current>& activeCollisions,
    CDiffContactGrad<kGradTarget>& outGrad) {
  if (activeCollisions.empty()) {
    return;
  }

  ContactEvalConfig config = InitializeContactForceAdjointEvalConfig(simParams.value);
  config = RefineContactForceAdjointEvalConfig(reg, e, config);

  // Gradient accumulators.
  Vec4r outGradientCom = {};
  Vec4r outGradientRot = {};

  // Match the forward async contact assembly path: use local FILO memory for the temporary
  // response.
  MOCHI_FILO_STACK_ALLOCATOR(allocator, 32 * 1024);
  CollisionResponseResult collisionResponse(&allocator);
  collisionResponse.Reserve(activeCollisions, false, true, true);

  auto const com = pose.value.VGetTranslation();
  for (auto& collision : activeCollisions) {
    ContactDetectionResult& contactQuery = collision.collisionResult;
    int const numContacts = isize(contactQuery.forceAdjoint);
    if (numContacts <= 0) {
      continue;
    }
    MOCHI_ASSERT_VERBOSE(isize(contactQuery.sampleIndices) == numContacts, "Unexpected size");

    // Compute the Jacobians of contact forces wrt contact positions.
    auto contactParams = GetContactPairParams(reg, e, collision.colliderEntity);
    collisionResponse.ResizeNoInit(numContacts, false, true, true);
    ComputeCollisionResponse<kGradTarget>(
        contactQuery,
        contactParams,
        config,
        intState.dtStage,
        false,
        false,
        true,
        collisionResponse);

    // Compute the gradient wrt contact positions. Reuse `force` for storage.
    for (int i = 0; i < numContacts; ++i) {
      collisionResponse.force[i] = ToReal3(
          DotVecMat3x3(ToSimd(contactQuery.forceAdjoint[i]), collisionResponse.dforce[i]));
    }

    // Accumulation of gradient terms for all contact points.
    // The implementation matches AssembleRigidBodyAsyncContactResponse.
    auto const& colliderTransform =
        reg.get<CRootTransform const>(collision.colliderEntity).worldFromLocalPrev;
    auto const trans = colliderTransform.VGetTranslation();
    auto const [rot, rotT] = ToVMatrix3x3_WithTranspose(colliderTransform.GetRotation());
    auto const comColliderSpace = DotVecMat3x3(com - trans, rot);

    Vec4r res = {};
    Vec4r skJRes = {};
    for (int i = 0; i < numContacts; ++i) {
      auto const posColliding =
          ToSimd(GetCollidingPosition<GetTimeStep<kGradTarget>()>(contactQuery, i));
      auto const jVec = posColliding - comColliderSpace;
      Vec4r const collRes = ToSimd(collisionResponse.force[i]);
      res += collRes;
      skJRes += Cross3(jVec, collRes);
    }

    outGradientCom += DotVecMat3x3(res, rotT);
    outGradientRot += DotVecMat3x3(skJRes, rotT);
  }

  ColumnVector<real, RigidSize::kDAll> grad;
  Store(grad.data(), outGradientCom);
  Store<RigidSize::kDRot>(grad.data() + RigidSize::kDTrans, outGradientRot);
  outGrad += grad;
}

// The force adjoints of a rigid actor's (or link's) samples against a soft-body SDF collider
// (a mapped collider: its rest-space grid SDF through the deformation of its tetrahedra). The
// queried world force is Sum_s w_s J_s^T f_s(p_s) with J_s = jacColliderFromWorld[s] the Jacobian
// of the mapping (rest space from the world) and p_s = map(x_s) the sample's rest-space position.
// - Through p_s: the colliding actor's DoFs move x_s (d p_s / d x_s = J_s: translation Sum g_s,
//   rotation Sum (x_s - c_A) x g_s with g_s = J_s^T collRes_s), and the collider's nodes move
//   the mapping (d p_s / d node_k = -J_s b_k I, with b_k the barycentric coordinates the query
//   stores in jacWorldFromDofs: -b_k g_s).
// - Through J_s (current state only): J_s = F^-1 with F = Sum_k x_k (x) G_k the deformation
//   gradient of the tetrahedron and G_k the rest-space gradients of its barycentric coordinates,
//   so d(J_s^T f_s)/d x_{k,a} = -(J_s^T G_k) (J_s^T f_s)_a and the adjoint carries
//   -(lambda_s . G_k) (f_world,s)_a with lambda_s the weighted collider-space adjoint.
// The previous-state target uses the Jacobians of the stage-start mapping.
template <GradTarget kGradTarget>
static void AccumulateRigidVsMappedColliderForceAdjoints(
    entt::registry& reg,
    entt::entity e,
    entt::entity e2,
    ContactDetectionResult const& query,
    CollisionResponseResult const& response, // force[s] holds lambda_s^T dforce_s (collRes)
    Span<Real3 const> forwardForces, // the forward forces per unit area (current target only)
    ColumnVectorView<real> outGradA) {
  TimeStep constexpr kTimeStep = GetTimeStep<kGradTarget>();
  bool constexpr kCurrent = kGradTarget == GradTarget::Current;
  int const numContacts = isize(query.forceAdjoint);
  auto const& jacs = kCurrent ? query.jacColliderFromWorld : query.jacColliderFromWorldStageStart;
  auto const& dofsJacs = kCurrent ? query.jacWorldFromDofs : query.jacWorldFromDofsStageStart;
  MOCHI_ASSERT(
      isize(jacs) == numContacts && isize(dofsJacs) == numContacts,
      "Missing per-contact mapping Jacobians of a soft-body SDF collider.");
  auto const& samples = reg.get<CContactSamples<TimeStep::Current> const>(e);
  auto const& rootA = reg.get<CRootTransform const>(e);
  TransformRT const& worldFromA = kCurrent ? rootA.worldFromLocal : rootA.worldFromLocalStageStart;
  Vec4r const comA = reg.get<CRigidState<kTimeStep> const>(e).value.VGetTranslation();
  auto outGradB = AsView(reg.get<CDiffContactGrad<kGradTarget>>(e2));
  TetrahedralMap const* map = nullptr;
  if constexpr (kCurrent) {
    map = dynamic_cast<TetrahedralMap const*>(
        reg.get<CSdfMapping<TimeStep::Current> const>(e2).get());
    MOCHI_ASSERT(map, "Contact-force adjoints need a tetrahedral mapping of the soft collider.");
  }
  int restGradientsTet = -1;
  VMatrix3x3r restGradients = {};
  Vec4r res = {};
  Vec4r skARes = {};
  for (int s = 0; s < numContacts; ++s) {
    Vec4r const collRes = ToSimd(response.force[s]);
    Vec4r const g = DotVecMat3x3(collRes, jacs[s]); // J_s^T collRes_s, in the world
    res += g;
    Vec4r const xs =
        worldFromA.TransformPoint(ToSimd(samples.positions[query.sampleIndices[s]], 1_r));
    skARes += Cross3(xs - comA, g);
    auto const& dofsJac = dofsJacs[s];
    for (int j = 0; j < 12; ++j) {
      outGradB[dofsJac.inds[j]] -= Get0(VDot<3>(g, dofsJac.jac[j]));
    }
    if constexpr (kCurrent) {
      MOCHI_ASSERT(dofsJac.element >= 0, "The mapping did not report the contact's tetrahedron.");
      if (dofsJac.element != restGradientsTet) {
        restGradientsTet = dofsJac.element;
        restGradients = map->RestBarycentricGradients(restGradientsTet);
      }
      Vec4r const lambda = ToSimd(query.forceAdjoint[s]);
      Real3 const fWorld = ToReal3(DotVecMat3x3(ToSimd(forwardForces[s]), jacs[s]));
      Vec4r const gradient3 = -(restGradients[0] + restGradients[1] + restGradients[2]);
      for (int k = 0; k < 4; ++k) {
        real const c = Get0(VDot<3>(lambda, k < 3 ? restGradients[k] : gradient3));
        for (int a = 0; a < 3; ++a) {
          outGradB[dofsJac.inds[3 * k + a]] -= c * fWorld[a];
        }
      }
    }
  }
  ColumnVector<real, RigidSize::kDAll> gradA;
  Store(gradA.data(), res);
  Store<RigidSize::kDRot>(gradA.data() + RigidSize::kDTrans, skARes);
  outGradA += gradA;
}

static bool IsMappedSoftCollider(entt::registry const& reg, entt::entity e) {
  return reg.all_of<TagSoftActor>(e) && !reg.any_of<TagNestedSoftActor, TagRomActor>(e) &&
      reg.try_get<CSdfMapping<TimeStep::Current> const>(e) != nullptr;
}

template <GradTarget kGradTarget>
static void AccumulateAllSyncRigidContactForceAdjoints(
    entt::registry& reg,
    Span<entt::entity const> actors) {
  MOCHI_PROFILE_SCOPE();

  ContactEvalConfig const configAllPairs =
      InitializeContactForceAdjointEvalConfig(reg.ctx<CSimulationParams const>());

  MOCHI_FILO_STACK_ALLOCATOR(allocator, 32 * 1024);

  for (auto e : actors) {
    // Only rigid-actor contact is supported in differentiability.
    if (!reg.all_of<TagRigidActor>(e)) {
      continue;
    }
    auto* activeCollisions =
        reg.try_get<CActiveCollisions<ContactType::Sync, TimeStep::Current>>(e);
    if (!activeCollisions) {
      continue;
    }

    // Params for this colliding actor
    real const dtStage = reg.get<CTimeIntegratorState const>(e).dtStage;
    ContactEvalConfig const configPair =
        RefineContactForceAdjointEvalConfig(reg, e, configAllPairs);

    // Target component for this colliding actor
    auto outGradA = AsView(reg.get<CDiffContactGrad<kGradTarget>>(e));

    // Reserve for the largest collision up-front so the per-collision ResizeNoInit below never
    // reallocates, which the FILO allocator requires.
    CollisionResponseResult response(&allocator);
    response.Reserve(*activeCollisions, false, true, true);

    // Traverse all its active collisions
    for (auto& coll : *activeCollisions) {
      // `forceAdjoint` holds the contact-force adjoints; it is sized only if contact queries were
      // enabled for some actor in the contact pair.
      auto& query = coll.collisionResult;
      int const numContacts = isize(query.forceAdjoint);
      if (numContacts <= 0) {
        continue;
      }
      MOCHI_ASSERT_VERBOSE(isize(query.sampleIndices) == numContacts, "Unexpected size");

      // Rigid colliders (standalone or links) and soft-body SDF colliders; others are refused
      // by the query backward.
      auto const e2 = coll.colliderEntity;
      bool const colliderIsRigid = reg.all_of<TagRigidActor>(e2);
      if (!colliderIsRigid && !IsMappedSoftCollider(reg, e2)) {
        continue;
      }

      // Compute the Jacobians of contact forces wrt contact positions (and, for the current
      // state, the forces themselves: see rotForceTerm).
      auto const contactParams = GetContactPairParams(reg, e, e2);
      bool constexpr kAssemForce = kGradTarget == GradTarget::Current;
      response.ResizeNoInit(numContacts, false, true, true);
      ComputeCollisionResponseRange<kGradTarget>(
          {0, numContacts},
          query,
          contactParams,
          configPair,
          dtStage,
          false,
          kAssemForce,
          true,
          response);
      // The queried world force is Sum_s w_s R_B f_s. With a dynamic collider B, the rotation
      // of B also turns the forces: for the left rotation increment delta of B's chart,
      // d(R_B f_s)/d delta = delta x R_B f_s, so the adjoint of lambda . F w.r.t. delta carries
      // Sum_s (R_B f_s) x (w_s lambda) = R_B Sum_s f_s x lambda_s, with lambda_s the weighted
      // collider-space adjoint held in forceAdjoint. The position derivatives below
      // (through p_s = R_B^T (x_s - t_B)) do not contain it. Accumulate it before `force` is
      // reused.
      Vec4r rotForceTerm = {};
      DynamicArray<Real3> forwardForces(&allocator);
      if constexpr (kAssemForce) {
        forwardForces.resize_noinit(numContacts);
        for (int s = 0; s < numContacts; ++s) {
          forwardForces[s] = response.force[s];
          rotForceTerm += Cross3(ToSimd(response.force[s]), ToSimd(query.forceAdjoint[s]));
        }
      }

      // Compute the gradient wrt contact positions. Reuse `force` for storage.
      for (int i = 0; i < numContacts; ++i) {
        response.force[i] =
            ToReal3(DotVecMat3x3(ToSimd(query.forceAdjoint[i]), response.dforce[i]));
      }
      if (!colliderIsRigid) {
        AccumulateRigidVsMappedColliderForceAdjoints<kGradTarget>(
            reg, e, e2, query, response, forwardForces, outGradA);
        continue;
      }

      // Accumulation of gradient terms for all contact points.
      // The implementation matches AssembleCollisionResponseRange_SyncRigid.
      TimeStep constexpr kTimeStep = GetTimeStep<kGradTarget>();
      Vec4r comA = reg.template get<CRigidState<kTimeStep> const>(e).value.VGetTranslation();
      auto const& stateB = reg.template get<CRigidState<kTimeStep> const>(e2).value;
      auto [rotB, rotBT] = ToVMatrix3x3_WithTranspose(stateB.GetRotation());
      Vec4r comB = stateB.VGetTranslation();
      Vec4r comBLocal = reg.template get<CRigidBodyInertia const>(e2).GetCenterOfMassLocal();

      Vec4r res = {};
      Vec4r skPRes = {};
      for (int s = 0; s < numContacts; ++s) {
        Vec4r const collRes = ToSimd(response.force[s]);
        res += collRes;
        auto posColliding = ToSimd(GetCollidingPosition<kTimeStep>(query, s));
        skPRes += Cross3(posColliding - comBLocal, collRes);
      }
      res = DotVecMat3x3(res, rotBT);
      skPRes = DotVecMat3x3(skPRes, rotBT);
      rotForceTerm = DotVecMat3x3(rotForceTerm, rotBT);

      // Target component for the collider actor
      auto outGradB = AsView(reg.get<CDiffContactGrad<kGradTarget>>(e2));

      ColumnVector<real, RigidSize::kDAll> gradA;
      ColumnVector<real, RigidSize::kDAll> gradB;
      Store(gradA.data(), res);
      Store<RigidSize::kDRot>(gradA.data() + RigidSize::kDTrans, skPRes - Cross3(comA - comB, res));
      Store(gradB.data(), -res);
      Store<RigidSize::kDRot>(gradB.data() + RigidSize::kDTrans, -skPRes + rotForceTerm);

      outGradA += gradA;
      outGradB += gradB;
    }
  }
}

// The force adjoints of a deformable actor's samples (the quadrature points of a soft body's
// boundary elements, of a shell's surface elements, of a rod's centerline segments) against a
// queried rigid collider. The samples' collider-space positions depend on the actor's nodal DoFs
// through the element interpolation and the rigid pose, the same map the contact assembly uses
// (deformable::SetupCollidingJacobians); it is rebuilt here for the pair at the target time step,
// since the registered contact Jacobians belong to the last assembly. The collider side is the
// rigid accumulation above (lever arms about the collider's center of mass, the rotation of the
// forces with the collider).
template <GradTarget kGradTarget, typename ActorTag, typename DiscretizationType, int kNumFields>
static void AccumulateAllSyncDeformableCollidingForceAdjoints(
    entt::registry& reg,
    Span<entt::entity const> actors) {
  MOCHI_PROFILE_SCOPE();
  TimeStep constexpr kTimeStep = GetTimeStep<kGradTarget>();
  bool constexpr kCurrent = kGradTarget == GradTarget::Current;
  ContactEvalConfig const configAllPairs =
      InitializeContactForceAdjointEvalConfig(reg.ctx<CSimulationParams const>());
  MOCHI_FILO_STACK_ALLOCATOR(allocator, 32 * 1024);
  for (auto e : actors) {
    if (!reg.all_of<ActorTag>(e) ||
        reg.any_of<TagNestedSoftActor, TagRomActor, TagRodSurfaceContact>(e)) {
      continue;
    }
    auto* activeCollisions =
        reg.try_get<CActiveCollisions<ContactType::Sync, TimeStep::Current>>(e);
    auto const* discretization = reg.try_get<DiscretizationType const>(e);
    if (!activeCollisions || !discretization) {
      continue;
    }
    real const dtStage = reg.get<CTimeIntegratorState const>(e).dtStage;
    ContactEvalConfig const configPair =
        RefineContactForceAdjointEvalConfig(reg, e, configAllPairs);
    auto const& rootA = reg.get<CRootTransform const>(e);
    int const dofOffsetA = reg.get<CDofOffset const>(e).dofsOffset;
    auto outGradA = AsView(reg.get<CDiffContactGrad<kGradTarget>>(e));
    CollisionResponseResult response(&allocator);
    response.Reserve(*activeCollisions, false, true, true);
    for (auto& coll : *activeCollisions) {
      auto& query = coll.collisionResult;
      int const numContacts = isize(query.forceAdjoint);
      if (numContacts <= 0) {
        continue;
      }
      MOCHI_ASSERT_VERBOSE(isize(query.sampleIndices) == numContacts, "Unexpected size");
      auto const e2 = coll.colliderEntity;
      if (!reg.all_of<TagRigidActor>(e2)) {
        continue; // queried actors are rigid; other colliders are refused by the query backward
      }
      // The samples' collider-space Jacobian w.r.t. the soft's DoFs at the target time step.
      auto const& rootB = reg.get<CRootTransform const>(e2);
      VMatrix3x3r const jacColliderFromWorld = ToVMatrix3x3Transpose(
          (kCurrent ? rootB.worldFromLocal : rootB.worldFromLocalStageStart).GetRotation());
      std::array<ContactJac, JacData::kMaxJacs> jacs;
      dmap::DMapDeformable<kNumFields> dsoft(0, dofOffsetA);
      dmap::DMapRTConst dtransform(kCurrent ? rootA.worldFromLocal : rootA.worldFromLocalStageStart);
      discretization->Visit([&](auto const& discretizationImpl) {
        using DiscretizationT = std::decay_t<decltype(discretizationImpl)>;
        using DQuad = dmap::DMapQuad<typename DiscretizationT::ElementT>;
        DQuad dquad(discretizationImpl.femElements, MakeSingletonConstSpan(jacColliderFromWorld));
        dmap::DMap<DQuad, dmap::DMapRTConst, dmap::DMapDeformable<kNumFields>> dmapPair(
            &dquad, &dtransform, &dsoft);
        dmapPair.GetJac(query.sampleIndices, jacs);
      });
      ContactJac const& jac = jacs[0];
      MOCHI_ASSERT(jac.nContacts == numContacts, "Unexpected contact Jacobian size.");
      // Jacobians of the contact forces wrt contact positions (and the forces themselves).
      auto const contactParams = GetContactPairParams(reg, e, e2);
      response.ResizeNoInit(numContacts, false, true, true);
      ComputeCollisionResponseRange<kGradTarget>(
          {0, numContacts},
          query,
          contactParams,
          configPair,
          dtStage,
          false,
          kCurrent,
          true,
          response);
      Vec4r rotForceTerm = {};
      if constexpr (kCurrent) {
        for (int s = 0; s < numContacts; ++s) {
          rotForceTerm += Cross3(ToSimd(response.force[s]), ToSimd(query.forceAdjoint[s]));
        }
      }
      for (int i = 0; i < numContacts; ++i) {
        response.force[i] =
            ToReal3(DotVecMat3x3(ToSimd(query.forceAdjoint[i]), response.dforce[i]));
      }
      // The soft's nodal DoFs, through the samples' Jacobian.
      for (int s = 0; s < numContacts; ++s) {
        auto const jacS = jac.Jac(s);
        auto const inds = jac.Inds(s);
        Real3 const& collRes = response.force[s];
        for (int j = 0; j < jac.nDoFsInternal; ++j) {
          auto const column = jacS.Col(j);
          outGradA[inds[j] - dofOffsetA] +=
              column[0] * collRes[0] + column[1] * collRes[1] + column[2] * collRes[2];
        }
      }
      // The rigid collider, as in the rigid-rigid accumulation.
      auto const& stateB = reg.template get<CRigidState<kTimeStep> const>(e2).value;
      auto [rotB, rotBT] = ToVMatrix3x3_WithTranspose(stateB.GetRotation());
      Vec4r comBLocal = reg.template get<CRigidBodyInertia const>(e2).GetCenterOfMassLocal();
      Vec4r res = {};
      Vec4r skPRes = {};
      for (int s = 0; s < numContacts; ++s) {
        Vec4r const collRes = ToSimd(response.force[s]);
        res += collRes;
        auto posColliding = ToSimd(GetCollidingPosition<kTimeStep>(query, s));
        skPRes += Cross3(posColliding - comBLocal, collRes);
      }
      res = DotVecMat3x3(res, rotBT);
      skPRes = DotVecMat3x3(skPRes, rotBT);
      rotForceTerm = DotVecMat3x3(rotForceTerm, rotBT);
      auto outGradB = AsView(reg.get<CDiffContactGrad<kGradTarget>>(e2));
      ColumnVector<real, RigidSize::kDAll> gradB;
      Store(gradB.data(), -res);
      Store<RigidSize::kDRot>(gradB.data() + RigidSize::kDTrans, -skPRes + rotForceTerm);
      outGradB += gradB;
    }
  }
}

template <GradTarget kGradTarget>
static void AccumulateIslandContactForceAdjoints(
    entt::registry& reg,
    CIslandDescendants const& descendants) {
  // Handle async contact per actor
  ecs::InvokeForEach(
      &AccumulateAsyncContactForceAdjoints<kGradTarget>, reg, descendants.rigidActors);

  // Handle sync contact per island
  AccumulateAllSyncRigidContactForceAdjoints<kGradTarget>(reg, descendants.rigidActors);
  AccumulateAllSyncDeformableCollidingForceAdjoints<
      kGradTarget,
      TagSoftActor,
      CFemBoundaryDiscretization,
      3>(reg, descendants.softActors);
  AccumulateAllSyncDeformableCollidingForceAdjoints<
      kGradTarget,
      TagShellActor,
      CFemSurfaceDiscretization,
      3>(reg, descendants.shellActors);
  AccumulateAllSyncDeformableCollidingForceAdjoints<
      kGradTarget,
      TagRodActor,
      CFemSegmentDiscretization,
      4>(reg, descendants.rodActors);
}

void mochi::AccumulateContactForceAdjoints(entt::registry& reg) {
  MOCHI_PROFILE_SCOPE();

  reg.view<CDiffContactGrad<GradTarget::Current>, CDiffContactGrad<GradTarget::Previous>>().each(
      [](CDiffContactGrad<GradTarget::Current>& outGradCurr,
         CDiffContactGrad<GradTarget::Previous>& outGradPrev) {
        outGradCurr.SetZero();
        outGradPrev.SetZero();
      });

  reg.view<CIslandDescendants const>().each([&](CIslandDescendants const& descendants) {
    AccumulateIslandContactForceAdjoints<GradTarget::Current>(reg, descendants);
    AccumulateIslandContactForceAdjoints<GradTarget::Previous>(reg, descendants);
  });
}
