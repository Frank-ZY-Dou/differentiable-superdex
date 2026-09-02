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

#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/vmatrix.h>

#include <array>

namespace mochi {
/**
 * @brief SDF and gradient for each sample point that contacted the collider
 */
struct SdfInfo {
  // PERFORMANCE: All arrays should contain POD types for fast resize and clear.

  /**
   * @brief Value of the signed distance.
   * @note Negative values are inside the collider.
   */
  DynamicArray<real> val = {};

  /**
   * @brief Gradient of the signed distance for each sample point in contact, expressed in the
   * collider's local frame.
   * @note 1-to-1 with val
   * @note Same as the collider's normal for exact SDFs, but it may not be unitary for inexact SDFs.
   * The collider's normal can always be recovered by normalizing the gradient of the SDF.
   */
  DynamicArray<Real3> grad = {};

  /**
   * @brief Optional: Hessian of the signed distance for each sample point in contact (the Jacobian
   * of @ref grad), expressed in the collider's local frame.
   * @note Filled only when requested with ContactDetectionParams::computeSdfHessian (then
   * @ref hasHessian is true and the array is 1-to-1 with val), otherwise empty. The adjoint of a
   * differentiable scene needs it at the stage start: with explicit normals the dissipative contact
   * terms use the stage-start SDF gradient as the contact normal, and the previous-state gradient
   * (GradTarget::Previous) must differentiate that normal w.r.t. the stage-start position.
   */
  DynamicArray<Matrix3x3r> hess = {};

  /** @brief True if @ref hess is maintained 1-to-1 with val. Set by the query that fills it. */
  bool hasHessian = false;

  /** @brief Construct with default allocator */
  SdfInfo() = default;

  /** @brief Construct with custom allocator */
  explicit SdfInfo(Allocator* allocator) : val(allocator), grad(allocator), hess(allocator) {}

  /** @brief Return true if there is no data */
  bool empty() const {
    CheckSizes();
    return val.empty();
  }

  /** @brief Return the size of the arrays */
  size_t size() const {
    CheckSizes();
    return val.size();
  }

  void reserve(size_t capacity) {
    val.reserve(capacity);
    grad.reserve(capacity);
    if (hasHessian) {
      hess.reserve(capacity);
    }
  }

  /** @brief Resize the arrays with default values */
  void resize(size_t size) {
    val.resize(size);
    grad.resize(size);
    if (hasHessian) {
      hess.resize(size);
    }
  }

  /** @brief Resize the arrays without initializing new elements */
  void resize_noinit(size_t size) {
    val.resize_noinit(size);
    grad.resize_noinit(size);
    if (hasHessian) {
      hess.resize_noinit(size);
    }
  }

  /** @brief Clear the contact info without releasing memory. Also drops the Hessian request. */
  void clear() {
    CheckSizes();
    val.clear();
    grad.clear();
    hess.clear();
    hasHessian = false;
  }

  /** @brief Push a new contact (only valid without Hessians) */
  MOCHI_FORCE_INLINE void push_back(real v, Real3 const& g) {
    MOCHI_ASSERT_VERBOSE(!hasHessian, "This SdfInfo carries Hessians: push them too.");
    val.push_back(v);
    grad.push_back(g);
  }

  /** @brief Push a new contact with its Hessian (only valid with Hessians) */
  MOCHI_FORCE_INLINE void push_back(real v, Real3 const& g, Matrix3x3r const& h) {
    MOCHI_ASSERT_VERBOSE(hasHessian, "This SdfInfo carries no Hessians.");
    val.push_back(v);
    grad.push_back(g);
    hess.push_back(h);
  }

  /** @brief Append all the contact info from another SdfInfo to this one */
  void append(SdfInfo const& rhs) {
    MOCHI_ASSERT(
        hasHessian == rhs.hasHessian, "Cannot append SdfInfo with a different Hessian request.");
    val.append(rhs.val);
    grad.append(rhs.grad);
    if (hasHessian) {
      hess.append(rhs.hess);
    }
    CheckSizes();
  }

 private:
  void CheckSizes() const {
    MOCHI_ASSERT_VERBOSE(grad.size() == val.size());
    MOCHI_ASSERT_VERBOSE(hasHessian ? (hess.size() == val.size()) : hess.empty());
  }
};

/*
  For deformable colliders, Jacobian of deformed position wrt dofs.
*/
struct ColliderJacDofs {
  // To avoid dynamic memory allocation, we set a maximum number of dofs.
  static constexpr int kMaxDoFs = 60;
  NdArray<Vec4r, kMaxDoFs> jac = {};
  std::array<int, kMaxDoFs> inds =
      {}; // Indices of the dofs (local to the actor, not global to the sim)
};

/*
  Parameters for contact detection
*/
struct ContactDetectionParams {
  // Tolerance value to use during contact detection
  real tolerance = 1e-3_r;
  // If true, acceleration structures will be used for detection.
  bool useAccelerationStructures = true;
  // If true, the query also returns the Hessian of the signed distance at each contact
  // (SdfInfo::hess). Supported by plane, sphere, box and grid-SDF colliders; mesh, point-cloud and
  // mapped colliders assert. Needed by the adjoint of differentiable scenes at the stage start.
  bool computeSdfHessian = false;
};

} // namespace mochi
