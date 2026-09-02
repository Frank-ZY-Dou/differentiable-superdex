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

#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/quaternion.h>
#include <mochi_core/utils/quaternion_utils.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/vmatrix.h>

#include <limits>

namespace mochi {

// Computes the Rodrigues rotation for some rotation vector p = axis * angle.
// Rodrigues(p) = exp(skew(p))
template <typename T>
[[nodiscard]] NdArray<Simd<T, 4>, 3> Rodrigues(Simd<T, 4> p) {
  // [p] skew-symmetric: [p][p]^T = -[p][p]
  // R = I + sinc(a)*[p] - .5*sinc^2(a/2)[p][p]^T

  NdArray<Simd<T, 4>, 3> S = Skew3(p);
  NdArray<Simd<T, 4>, 3> S2neg = Dot3x3(S, Transpose3x3(S));
  T angle = Get0(VNorm<3>(p));
  T sinchalf = Sinc(angle / T(2));

  return VEye<3, T>() + S * Sinc(angle) - T(0.5) * sinchalf * sinchalf * S2neg;
}

// Computes the rotation vector corresponding to a rotation matrix.
// InvRodrigues(R) = skewInv(log(R))
[[nodiscard]] Vec4r InvRodrigues(VMatrix3x3r const& R);

// Cap a rotation vector to a norm <= pi.
[[nodiscard]] Vec4r RotVectorPiCap(Vec4r rotVec);

[[nodiscard]] Real3 RotVectorPiCap(Real3 rotVec);

// Given a rotation r in rotation vector representation, an incremental change of the rotation
// can be expressed as R(theta) * R(r) = R(r + dr), for a differential rotation vector theta -> 0.
// The following functions compute the partial derivatives dr/dtheta and dtheta/dr.

namespace drotvector {

// Accurate methods for large r.
// dtheta/dr = inv(sk(r) + r rT) * (R(r) - eye + r rT).
template <typename T>
[[nodiscard]] NdArray<Simd<T, 4>, 3> DThetaDLargeR(Simd<T, 4> r, NdArray<Simd<T, 4>, 3> const& R) {
  MOCHI_ASSERT_VERBOSE(NearEqual(Rodrigues(r), R, Simd<T, 4>(T(1e-5))), "Inconsistent inputs");
  NdArray<Simd<T, 4>, 3> skew = Skew3(r);
  NdArray<Simd<T, 4>, 3> outer = Outer3(r, r);
  return Dot3x3(Invert3x3(skew + outer), R - VEye<3, T>() + outer);
}
// dr/dtheta = inv(R(r) - eye + r rT) * (sk(r) + r rT).
template <typename T>
[[nodiscard]] NdArray<Simd<T, 4>, 3> DLargeRDTheta(Simd<T, 4> r, NdArray<Simd<T, 4>, 3> const& R) {
  MOCHI_ASSERT_VERBOSE(NearEqual(Rodrigues(r), R, Simd<T, 4>(T(1e-5))), "Inconsistent inputs");
  NdArray<Simd<T, 4>, 3> skew = Skew3(r);
  NdArray<Simd<T, 4>, 3> outer = Outer3(r, r);
  return Dot3x3(Invert3x3(R - VEye<3, T>() + outer), skew + outer);
}

// Series methods for small r (the closed forms above divide by quantities that vanish with r).
// With t = |r|^2, the left Jacobian of SO(3) and its inverse are
//   dtheta/dr = eye + a sk(r) + b sk2(r),  a = (1 - cos|r|) / |r|^2 = 1/2 - t/24 + t^2/720 - ...
//                                          b = (|r| - sin|r|) / |r|^3 = 1/6 - t/120 + t^2/5040 - ...
//   dr/dtheta = eye - 1/2 sk(r) + c sk2(r), c = 1/12 + t/720 + t^2/30240 + ...
// Truncated after t^2, the series are accurate to about t^3 / 1e5 (1e-15 at the switching
// thresholds below), i.e. to double-precision roundoff; the former first-order forms
// (eye +- 0.5 sk(r)) had an O(t) error of up to 5e-5 there, which was visible in double-precision
// gradient checks (diffsim::SetVelocityBackward).
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3> DThetaDSmallR(Simd<T, 4> r) {
  T const t = NormSqr<3>(r);
  T const a = T(0.5) - t * (T(1) / T(24)) + t * t * (T(1) / T(720));
  T const b = T(1) / T(6) - t * (T(1) / T(120)) + t * t * (T(1) / T(5040));
  NdArray<Simd<T, 4>, 3> const skew = Skew3(r);
  return VEye<3, T>() + a * skew + b * Dot3x3(skew, skew);
}
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3> DSmallRDTheta(Simd<T, 4> r) {
  T const t = NormSqr<3>(r);
  T const c = T(1) / T(12) + t * (T(1) / T(720)) + t * t * (T(1) / T(30240));
  NdArray<Simd<T, 4>, 3> const skew = Skew3(r);
  return VEye<3, T>() - T(0.5) * skew + c * Dot3x3(skew, skew);
}

// Default thresholds (on |r|^2) for choosing the closed forms or the series. The series' truncation
// error at the thresholds is at double-precision roundoff, and in single precision the series is at
// least as accurate as the closed forms up to there (validated in Rodrigues.DRotVectorThresholds;
// the float crossover where the closed forms take over lies far above, around |r|^2 = 0.2).
template <typename T>
static constexpr T kThresholdDRotIncrementDRotVector = T(0.0003227);
template <typename T>
static constexpr T kThresholdDRotVectorDRotIncrement = T(0.0003437);
} // namespace drotvector

// General methods, redirecting to the appropriate method depending on the size of r.
// Methods that take a quaternion are not templatized and must use the default type 'real'.
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3> DRotIncrementDRotVector(
    Simd<T, 4> r,
    NdArray<Simd<T, 4>, 3> const& R,
    T threshold = drotvector::kThresholdDRotIncrementDRotVector<T>) {
  return (NormSqr<3>(r) < threshold) ? drotvector::DThetaDSmallR(r)
                                     : drotvector::DThetaDLargeR(r, R);
}
[[nodiscard]] MOCHI_FORCE_INLINE VMatrix3x3r DRotIncrementDRotVector(
    Vec4r r,
    Quaternion const& q,
    real threshold = drotvector::kThresholdDRotIncrementDRotVector<real>) {
  return (NormSqr<3>(r) < threshold) ? drotvector::DThetaDSmallR(r)
                                     : drotvector::DThetaDLargeR(r, ToVMatrix3x3(q));
}
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3> DRotIncrementDRotVector(
    Simd<T, 4> r,
    T threshold = drotvector::kThresholdDRotIncrementDRotVector<T>) {
  return (NormSqr<3>(r) < threshold) ? drotvector::DThetaDSmallR(r)
                                     : drotvector::DThetaDLargeR(r, Rodrigues(r));
}
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3> DRotVectorDRotIncrement(
    Simd<T, 4> r,
    NdArray<Simd<T, 4>, 3> const& R,
    T threshold = drotvector::kThresholdDRotVectorDRotIncrement<T>) {
  return (NormSqr<3>(r) < threshold) ? drotvector::DSmallRDTheta(r)
                                     : drotvector::DLargeRDTheta(r, R);
}
[[nodiscard]] MOCHI_FORCE_INLINE VMatrix3x3r DRotVectorDRotIncrement(
    Vec4r r,
    Quaternion const& q,
    real threshold = drotvector::kThresholdDRotVectorDRotIncrement<real>) {
  return (NormSqr<3>(r) < threshold) ? drotvector::DSmallRDTheta(r)
                                     : drotvector::DLargeRDTheta(r, ToVMatrix3x3(q));
}
template <typename T>
[[nodiscard]] MOCHI_FORCE_INLINE NdArray<Simd<T, 4>, 3> DRotVectorDRotIncrement(
    Simd<T, 4> r,
    T threshold = drotvector::kThresholdDRotVectorDRotIncrement<T>) {
  return (NormSqr<3>(r) < threshold) ? drotvector::DSmallRDTheta(r)
                                     : drotvector::DLargeRDTheta(r, Rodrigues(r));
}

// Given two rotations qa, qb, and a reference frame q0, in quaternion representation, computes the
// relative rotation between them, Rab, in the R0 frame. This results in Rb * R0 = Ra * R0 * Rab
// => Rab = R0' * Ra' * Rb * R0.
[[nodiscard]] Quaternion RelativeRotation_Reference(Quaternion qa, Quaternion qb, Quaternion q0);

} // namespace mochi
