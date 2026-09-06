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

#include "dskinning.h"

#include <mochi_core/linear_algebra/krylov_interop.h>
#include <mochi_core/utils/dtransform.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/lie.h>
#include <mochi_core/utils/task_scheduler.h>
#include <mochi_core/utils/transform_rt_utils.h>

#include <utility>

namespace mochi {
namespace details {

constexpr int kMaxStackBones = 64;
constexpr size_t kBoneJacobiansStackSize =
    kMaxStackBones * sizeof(VMatrix3x3r) + alignof(VMatrix3x3r);
constexpr size_t kBoneParameterJacobiansStackSize =
    kMaxStackBones * (sizeof(VMatrix3x3r) + sizeof(VMatrix4x4r)) + alignof(VMatrix3x3r) +
    alignof(VMatrix4x4r);

template <bool kTranspose>
void ComputeBoneJacobians(
    DSkinningTransform const& dskinning,
    Span<TransformRT const> boneTransforms,
    DynamicArray<VMatrix3x3r>& jacobians) {
  jacobians.reserve(dskinning.GetBoneCount());
  for (int boneId = 0; boneId < dskinning.GetBoneCount(); ++boneId) {
    auto const& pre = dskinning.GetBonePreTransform(boneId);
    auto const& post = dskinning.GetBonePostTransform(boneId);
    real const scale = pre.GetScale() * post.GetScale();
    Quaternion const rot =
        post.GetRotation() * boneTransforms[boneId].GetRotation() * pre.GetRotation();
    if constexpr (kTranspose) {
      jacobians.emplace_back(scale * ToVMatrix3x3Transpose(rot));
    } else {
      jacobians.emplace_back(scale * ToVMatrix3x3(rot));
    }
  }
}

MOCHI_FORCE_INLINE VMatrix3x3r WeightedVertexJacobian(
    DSkinningTransform::VertexBones const& bones,
    Span<VMatrix3x3r const> jacobians) {
  auto vertexBones = MakeConstSpan(bones);
  MOCHI_ASSERT_VERBOSE(!vertexBones.empty(), "Vertex has no skinning bones.");
  VMatrix3x3r weightedJacobian = vertexBones[0].second * jacobians[vertexBones[0].first];
  for (auto const& [boneId, weight] : vertexBones.subspan(1)) {
    weightedJacobian += weight * jacobians[boneId];
  }
  return weightedJacobian;
}

// Precompute per-bone derivatives:
// - Translation derivatives are independent of the input and directly precomputed.
// - Rotation derivatives require the products of matrix transforms, which are precomputed.
template <bool kTransposePostMat>
void ComputeBoneParameterJacobians(
    DSkinningTransform const& dskinning,
    Span<TransformRT const> boneTransforms,
    DynamicArray<VMatrix3x3r>& postMat,
    DynamicArray<VMatrix4x4r>& preMatT) {
  postMat.reserve(dskinning.GetBoneCount());
  preMatT.reserve(dskinning.GetBoneCount());
  for (int boneId = 0; boneId < dskinning.GetBoneCount(); ++boneId) {
    auto const& pre = dskinning.GetBonePreTransform(boneId);
    auto const& post = dskinning.GetBonePostTransform(boneId);
    Quaternion const& boneRotation = boneTransforms[boneId].GetRotation();

    if constexpr (kTransposePostMat) {
      postMat.emplace_back(post.GetScale() * ToVMatrix3x3Transpose(post.GetRotation()));
    } else {
      postMat.emplace_back(post.GetScale() * ToVMatrix3x3(post.GetRotation()));
    }

    TransformSRT const bonePreTransform{
        pre.GetScale(), boneRotation * pre.GetRotation(), boneRotation * pre.VGetTranslation()};
    preMatT.emplace_back(ToVMatrix4x4Transpose(bonePreTransform));
  }
}

} // namespace details

void DSkinningTransform::Transform(
    Span<TransformRT const> boneTransforms,
    ColumnVectorView<real const> input,
    ColumnVectorView<real> output,
    Span<int const> activeVertices) const {
  MOCHI_PROFILE_SCOPE();
  static_assert(RigidSize::kDim == 3, "Only supported for 3D");
  MOCHI_ASSERT_VERBOSE(boneTransforms.size() == GetBoneCount());
  MOCHI_ASSERT_VERBOSE(input.Rows() % RigidSize::kDim == 0 && input.Rows() == output.Rows());

  // Precompute per-bone full transform and store its transpose to later operate with DotVecMat,
  // which is faster than DotMatVec.
  MOCHI_FILO_STACK_ALLOCATOR(alloc, 4096); // Capacity for 64 transforms with 32-bit floats.
  DynamicArray<VMatrix4x4r> transformT(&alloc);
  transformT.reserve(GetBoneCount());
  for (int boneId = 0; boneId < GetBoneCount(); ++boneId) {
    auto pre = ToVMatrix4x4(GetBonePreTransform(boneId));
    auto bone = ToVMatrix4x4(boneTransforms[boneId]);
    auto post = ToVMatrix4x4(GetBonePostTransform(boneId));
    transformT.emplace_back(Transpose4x4(Dot4x4(post, Dot4x4(bone, pre))));
  }

  auto workerTask = [&](int loopStart, int loopEnd) {
    constexpr int kDim = RigidSize::kDim;
    // This loop could be slightly faster by using full SIMD loads and stores for all vertices
    // except the last one, and partial SIMD loads and stores for the last one.
    for (int i = loopStart; i < loopEnd; ++i) {
      int const vertexId = activeVertices.empty() ? i : activeVertices[i];
      auto vertexBones = MakeConstSpan(perVertexBones[vertexId]);
      MOCHI_ASSERT_VERBOSE(!vertexBones.empty(), "Vertex has no skinning bones.");
      VMatrix4x4r weightedTransformT = vertexBones[0].second * transformT[vertexBones[0].first];
      for (auto const& [boneId, weight] : vertexBones.subspan(1)) {
        weightedTransformT += weight * transformT[boneId];
      }
      Vec4r inPoint = ToSimdPoint(Load<kDim, Vec4r>(&input[vertexId * kDim]));
      Store<kDim>(&output[vertexId * kDim], DotVecMat4x4(inPoint, weightedTransformT));
    }
  };

  constexpr int kMinVerticesPerTask = 6000; // 30 μs @ 5 ns per vertex (empirical value).
  int const numVertices =
      activeVertices.empty() ? input.Rows() / RigidSize::kDim : isize(activeVertices);
  ParallelForRange("Transform", 0, numVertices, kMinVerticesPerTask, INT_MAX, workerTask);
}

void DSkinningTransform::DTransform(
    Span<TransformRT const> boneTransforms,
    ColumnVectorView<real const> input,
    ColumnVectorView<real> output,
    Span<int const> activeVertices) const {
  MOCHI_PROFILE_SCOPE();
  static_assert(RigidSize::kDim == 3, "Only supported for 3D");
  MOCHI_ASSERT_VERBOSE(boneTransforms.size() == GetBoneCount());
  MOCHI_ASSERT_VERBOSE(input.Rows() % RigidSize::kDim == 0 && input.Rows() == output.Rows());

  // Store transposed per-bone Jacobians to use the faster DotVecMat3x3 in the vertex loop.
  MOCHI_FILO_STACK_ALLOCATOR(alloc, details::kBoneJacobiansStackSize);
  DynamicArray<VMatrix3x3r> jacobianT(&alloc);
  details::ComputeBoneJacobians</* kTranspose */ true>(*this, boneTransforms, jacobianT);
  Span<VMatrix3x3r const> const jacobianTSpan = MakeConstSpan(jacobianT);

  auto workerTask = [&](int loopStart, int loopEnd) {
    constexpr int kDim = RigidSize::kDim;
    for (int i = loopStart; i < loopEnd; ++i) {
      int const vertexId = activeVertices.empty() ? i : activeVertices[i];
      VMatrix3x3r const weightedJacobianT =
          details::WeightedVertexJacobian(perVertexBones[vertexId], jacobianTSpan);
      Vec4r const inVector = Load<kDim, Vec4r>(&input[vertexId * kDim]);
      Store<kDim>(&output[vertexId * kDim], DotVecMat3x3(inVector, weightedJacobianT));
    }
  };

  // Use the forward-transform threshold until this overload is benchmarked independently.
  constexpr int kMinVerticesPerTask = 6000;
  int const numVertices =
      activeVertices.empty() ? input.Rows() / RigidSize::kDim : isize(activeVertices);
  ParallelForRange("DTransformVector", 0, numVertices, kMinVerticesPerTask, INT_MAX, workerTask);
}

void DSkinningTransform::DTransform(
    Span<TransformRT const> boneTransforms,
    RowMatrixView<real const, krylov::kDynamic, RigidSize::kDim> inputJacobian,
    RowMatrixView<real, krylov::kDynamic, RigidSize::kDim> outputJacobian,
    Span<int const> activeVertices) const {
  MOCHI_PROFILE_SCOPE();
  static_assert(RigidSize::kDim == 3, "Only supported for 3D");
  MOCHI_ASSERT_VERBOSE(boneTransforms.size() == GetBoneCount());
  MOCHI_ASSERT_VERBOSE(inputJacobian.Rows() % RigidSize::kDim == 0);
  MOCHI_ASSERT_VERBOSE(inputJacobian.Rows() == outputJacobian.Rows());

  MOCHI_FILO_STACK_ALLOCATOR(alloc, details::kBoneJacobiansStackSize);
  DynamicArray<VMatrix3x3r> jacobians(&alloc);
  details::ComputeBoneJacobians</* kTranspose */ false>(*this, boneTransforms, jacobians);
  Span<VMatrix3x3r const> const jacobiansSpan = MakeConstSpan(jacobians);

  auto workerTask = [&](int loopStart, int loopEnd) {
    constexpr int kDim = RigidSize::kDim;
    VMatrix3x3r inVertexJac;
    for (int i = loopStart; i < loopEnd; ++i) {
      int const vertexId = activeVertices.empty() ? i : activeVertices[i];
      VMatrix3x3r const weightedJacobian =
          details::WeightedVertexJacobian(perVertexBones[vertexId], jacobiansSpan);
      LoadMatrix<3, 3>(inVertexJac, &inputJacobian(vertexId * kDim, 0));
      StoreMatrix<3, 3>(&outputJacobian(vertexId * kDim, 0), Dot3x3(weightedJacobian, inVertexJac));
    }
  };

  constexpr int kMinFlopsPerTask = 75000; // 50 μs @ 1.5 GFLOPs (small matrix operations).
  constexpr int kFlopsPerVertex =
      (2 * RigidSize::kDim - 1) * RigidSize::kDim * RigidSize::kDim; // Lower bound.
  constexpr int kMinVerticesPerTask = Max(1, kMinFlopsPerTask / kFlopsPerVertex);
  int const numVertices =
      activeVertices.empty() ? inputJacobian.Rows() / RigidSize::kDim : isize(activeVertices);
  ParallelForRange("DTransform", 0, numVertices, kMinVerticesPerTask, INT_MAX, workerTask);
}

void DSkinningTransform::DTransformDBonesTimesVector(
    Span<TransformRT const> boneTransforms,
    ColumnVectorView<real const> unposedPositions,
    ColumnVectorView<real const> input,
    ColumnVectorView<real> output,
    Span<int const> activeVertices) const {
  MOCHI_PROFILE_SCOPE();
  static_assert(RigidSize::kDim == 3, "Only supported for 3D");
  MOCHI_ASSERT_VERBOSE(boneTransforms.size() == GetBoneCount());
  MOCHI_ASSERT_VERBOSE(
      unposedPositions.Rows() % RigidSize::kDim == 0 && unposedPositions.Rows() == output.Rows());
  MOCHI_ASSERT_VERBOSE(input.Rows() == GetBoneCount() * RigidSize::kDAll);

  MOCHI_FILO_STACK_ALLOCATOR(alloc, details::kBoneParameterJacobiansStackSize);
  DynamicArray<VMatrix3x3r> postMatT(&alloc); // postMatT = (spost * Rpost)^T
  DynamicArray<VMatrix4x4r> preMatT(&alloc); // preMatT = (R * (spre Rpre, tpre))^T
  details::ComputeBoneParameterJacobians</* kTransposePostMat */ true>(
      *this, boneTransforms, postMatT, preMatT);

  auto workerTask = [&](int loopBegin, int loopEnd) {
    for (int i = loopBegin; i < loopEnd; ++i) {
      int const vertexId = activeVertices.empty() ? i : activeVertices[i];
      Vec4r const unposedPosition =
          ToSimdPoint(Load<RigidSize::kDim, Vec4r>(&unposedPositions[vertexId * RigidSize::kDim]));
      Vec4r result = {};
      for (auto const& [boneId, weight] : perVertexBones[vertexId]) {
        int const boneOffset = boneId * RigidSize::kDAll;
        Vec4r const translationVelocity = Load<RigidSize::kDTrans, Vec4r>(&input[boneOffset]);
        Vec4r const angularVelocity =
            Load<RigidSize::kDRot, Vec4r>(&input[boneOffset + RigidSize::kDTrans]);
        Vec4r const transformedPoint = DotVecMat4x4(unposedPosition, preMatT[boneId]);
        Vec4r const velocity = translationVelocity + Cross3(angularVelocity, transformedPoint);
        result += weight * DotVecMat3x3(velocity, postMatT[boneId]);
      }
      Store<RigidSize::kDim>(&output[vertexId * RigidSize::kDim], result);
    }
  };

  constexpr int kMinFlopsPerTask = 100000; // 20 μs @ 5 GFLOPs (SIMD operations).
  // Lower bound for one bone influence: matrix-vector product, cross product, and weighted
  // accumulation.
  constexpr int kMatrixVectorFlopsPerComponent = 2 * RigidSize::kDim - 1;
  constexpr int kCrossProductFlopsPerComponent = 3;
  constexpr int kWeightedAccumulationFlopsPerComponent = 2;
  constexpr int kFlopsPerVertex = RigidSize::kDim *
      (kMatrixVectorFlopsPerComponent + kCrossProductFlopsPerComponent +
       kWeightedAccumulationFlopsPerComponent);
  constexpr int kMinVerticesPerTask = Max(1, kMinFlopsPerTask / kFlopsPerVertex);
  int const numVertices =
      activeVertices.empty() ? unposedPositions.Rows() / RigidSize::kDim : isize(activeVertices);
  ParallelForRange(
      "DTransformDBonesTimesVector", 0, numVertices, kMinVerticesPerTask, INT_MAX, workerTask);
}

void DSkinningTransform::DTransformDBones(
    Span<TransformRT const> boneTransforms,
    ColumnVectorView<real const, krylov::kDynamic> input,
    SparseMatrix<real>& outputDBones,
    Span<int const> activeVertices) const {
  // NOTE: This method could be implemented using SIMD matrices instead of linear algebra library
  // matrices. The former is slightly faster but requires introducing overloads for the transpose
  // and matrix-matrix product of rectangular SIMD matrices that would be error-prone.
  MOCHI_PROFILE_SCOPE();
  static_assert(RigidSize::kDim == 3, "Only supported for 3D");
  MOCHI_ASSERT_VERBOSE(boneTransforms.size() == GetBoneCount());
  MOCHI_ASSERT_VERBOSE(input.Rows() % RigidSize::kDim == 0 && input.Rows() == outputDBones.Rows());

  MOCHI_FILO_STACK_ALLOCATOR(alloc, details::kBoneParameterJacobiansStackSize);
  DynamicArray<VMatrix3x3r> postMat(&alloc); // postMat = spost * Rpost
  DynamicArray<VMatrix4x4r> preMatT(&alloc); // preMatT = (R * (spre Rpre, tpre))^T
  details::ComputeBoneParameterJacobians</* kTransposePostMat */ false>(
      *this, boneTransforms, postMat, preMatT);

  auto workerTask = [&](int loopBegin, int loopEnd) {
    for (int i = loopBegin; i < loopEnd; ++i) {
      int const vertexId = activeVertices.empty() ? i : activeVertices[i];
      auto outRowValues = outputDBones.Values(vertexId * RigidSize::kDim);
      MOCHI_ASSERT_VERBOSE(
          isize(outRowValues) == outputDBones.Values(vertexId * RigidSize::kDim + 1).size() &&
          isize(outRowValues) == outputDBones.Values(vertexId * RigidSize::kDim + 2).size() &&
          isize(outRowValues) == RigidSize::kDAll * isize(perVertexBones[vertexId]));

      Vec4r inPoint = ToSimdPoint(Load<RigidSize::kDim, Vec4r>(&input[vertexId * RigidSize::kDim]));
      int colOffset = 0;
      for (auto const& [boneId, weight] : perVertexBones[vertexId]) {
        // Directly operate on the RigidSize::kDim x RigidSize::kDAll block in the output sparse
        // matrix that corresponds to the current (vertex, bone) pair.
        RowMatrixView<real, RigidSize::kDim, RigidSize::kDAll, krylov::kDynamic> outBlock(
            outRowValues.data() + colOffset,
            RigidSize::kDim,
            RigidSize::kDAll,
            isize(outRowValues));

        // Derivatives w.r.t. translation parameters.
        outBlock.template LeftCols<RigidSize::kDTrans>(RigidSize::kDTrans) =
            weight * AsMatrixView(postMat[boneId]);

        // Derivatives w.r.t. rotation parameters.
        auto inPointTransformed = weight * DotVecMat4x4(inPoint, preMatT[boneId]);
        outBlock.template RightCols<RigidSize::kDRot>(RigidSize::kDRot) =
            AsMatrixView(lie::DMultMatRotVecDRot(postMat[boneId], inPointTransformed));

        colOffset += RigidSize::kDAll;
      }
    }
  };

  constexpr int kMinFlopsPerTask = 200000; // 40 μs @ 5 GFLOPs (small matrix operations).
  constexpr int kFlopsPerVertex = RigidSize::kDim *
      (RigidSize::kDRot * (2 * RigidSize::kDim + 1) + RigidSize::kDTrans); // Lower bound.
  constexpr int kMinVerticesPerTask = Max(1, kMinFlopsPerTask / kFlopsPerVertex);
  int const numVertices =
      activeVertices.empty() ? input.Rows() / RigidSize::kDim : isize(activeVertices);
  ParallelForRange("DTransformDBones", 0, numVertices, kMinVerticesPerTask, INT_MAX, workerTask);
}

SparseMatrix<real> DSkinningTransform::CreateDBones() const {
  constexpr int kNumParams = RigidSize::kDAll;

  DynamicArray<int> cols;
  DynamicArray<int> ptr;
  cols.reserve(totalPairs * kNumParams * kDSkinningDofsPerVertex);
  ptr.reserve(kDSkinningDofsPerVertex * GetNumVertices() + 1);
  ptr.push_back(0);
  for (int vertexId = 0; vertexId < GetNumVertices(); ++vertexId) {
    for (int lr = 0; lr < kDSkinningDofsPerVertex; ++lr) {
      for (auto const& [boneId, weight] : perVertexBones[vertexId]) {
        for (int paramId = 0; paramId < kNumParams; ++paramId) {
          cols.push_back(kNumParams * boneId + paramId);
        }
      }
      ptr.push_back(isize(cols));
    }
  }

  return {boneCount * kNumParams, Graph<int, int>(std::move(ptr), std::move(cols))};
}

} // namespace mochi
