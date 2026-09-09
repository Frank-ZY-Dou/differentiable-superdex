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

#include <mochi_core/linear_algebra/krylov/incomplete_cholesky_prec.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/test/linear_algebra_test_helpers.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/task_scheduler.h>

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <vector>

using namespace mochi;
using namespace mochi::krylov;

// Use real instead of float or double to reduce build time. Both are checked by CI.
using Scalar = real;

static RowMatrix<real, 6, 6> const A{
    2.0_r,  0.0_r, -1.5_r, 0.0_r,  0.0_r, -1.2_r, 0.0_r,  3.0_r,  0.0_r, 0.0_r, 0.0_r,  -0.5_r,
    -1.5_r, 0.0_r, 4.0_r,  0.0_r,  0.0_r, 0.0_r,  0.0_r,  0.0_r,  0.0_r, 2.0_r, -1.0_r, 0.0_r,
    0.0_r,  0.0_r, 0.0_r,  -1.0_r, 2.0_r, -1.0_r, -1.2_r, -0.5_r, 0.0_r, 0.0_r, -1.0_r, 2.0_r};

//--- Incomplete Cholesky factor of A (from Octave)
static RowMatrix<real, 6, 6> R0{
    Sqrt(2.0_r),
    0.0_r,
    -Sqrt(1.125_r),
    0.0_r,
    0.0_r,
    -Sqrt(0.72_r),
    0.0_r,
    Sqrt(3.0_r),
    0.0_r,
    0.0_r,
    0.0_r,
    -Sqrt(0.25_r / 3.0_r),
    0.0_r,
    0.0_r,
    Sqrt(2.875_r),
    0.0_r,
    0.0_r,
    0.0_r,
    0.0_r,
    0.0_r,
    0.0_r,
    Sqrt(2.0_r),
    -1.0_r / Sqrt(2.0_r),
    0.0_r,
    0.0_r,
    0.0_r,
    0.0_r,
    0.0_r,
    Sqrt(1.5_r),
    -Sqrt(2.0_r / 3.0_r),
    0.0_r,
    0.0_r,
    0.0_r,
    0.0_r,
    0.0_r,
    Sqrt(0.53_r)};

//--- Cholesky factor of A (from Octave)
static RowMatrix<real, 6, 6> R{
    Sqrt(2.0_r),
    0.0_r,
    -Sqrt(1.125_r),
    0.0_r,
    0.0_r,
    -Sqrt(0.72_r),
    0.0_r,
    Sqrt(3.0_r),
    0.0_r,
    0.0_r,
    0.0_r,
    -Sqrt(0.25_r / 3.0_r),
    0.0_r,
    0.0_r,
    Sqrt(2.875_r),
    0.0_r,
    0.0_r,
    -0.5307910421576296_r,
    0.0_r,
    0.0_r,
    0.0_r,
    Sqrt(2.0_r),
    -1.0_r / Sqrt(2.0_r),
    0.0_r,
    0.0_r,
    0.0_r,
    0.0_r,
    0.0_r,
    Sqrt(1.5_r),
    -Sqrt(2.0_r / 3.0_r),
    0.0_r,
    0.0_r,
    0.0_r,
    0.0_r,
    0.0_r,
    0.4982578344243243_r};

//--- Singular matrix to test shifting
static RowMatrix<real, 5, 5> const M{1.0_r,  -1.0_r, 0.0_r,  0.0_r,  0.0_r,  -1.0_r, 2.0_r,
                                     -1.0_r, 0.0_r,  0.0_r,  0.0_r,  -1.0_r, 2.0_r,  -1.0_r,
                                     0.0_r,  0.0_r,  0.0_r,  -1.0_r, 2.0_r,  -1.0_r, 0.0_r,
                                     0.0_r,  0.0_r,  -1.0_r, 1.0_r};
//--- Cholesky factor of M (from Octave)
static RowMatrix<real, 5, 5> U{
    Sqrt(2.0_r),        -Sqrt(0.5_r),          0.0_r, 0.0_r, 0.0_r, 0.0_r,
    Sqrt(2.5_r),        -Sqrt(0.4_r),          0.0_r, 0.0_r, 0.0_r, 0.0_r,
    Sqrt(2.6_r),        -0.6201736729460422_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r,
    1.61721508012528_r, -0.6183469424008423_r, 0.0_r, 0.0_r, 0.0_r, 0.0_r,
    1.271867547672921_r};

template <int kBlockSize>
using BlockPrec = mochi::krylov::IncompleteCholeskyPrec<BlockSparseMatrix<real, kBlockSize>>;

template <typename MatrixType, int kBlockSize>
void CheckMultiRhsMatchesColumnSolves(
    BlockPrec<kBlockSize> const& prec,
    int size,
    int seed,
    real tolerance) {
  MatrixType x(size, 3), result(size, 3);
  x.SetRandom(seed);
  prec(x, result);
  for (int column = 0; column < x.Cols(); ++column) {
    ColumnVector<real> xColumn(size), expected(size);
    xColumn = x.Col(column);
    prec(xColumn, expected);
    EXPECT_TRUE(mochi::test::NearEqualMatrices(result.Col(column), expected, tolerance));
  }
}

template <int kBlockSize>
void CheckBlockPrecMultiRhs() {
  constexpr int kBlocks = 4;
  constexpr int kSize = kBlocks * kBlockSize;
  auto dense = RowMatrix<real, kSize, kSize>::Zero();
  for (int i = 0; i < kSize; ++i) {
    dense(i, i) = 4.0_r + real(i) * 0.05_r;
  }
  for (int block = 0; block < kBlocks; ++block) {
    int const next = (block + 1) % kBlocks;
    for (int i = 0; i < kBlockSize; ++i) {
      dense(block * kBlockSize + i, next * kBlockSize + i) = -0.2_r;
      dense(next * kBlockSize + i, block * kBlockSize + i) = -0.2_r;
    }
  }

  auto physical = ToBlockSparseMatrix<kBlockSize>(dense, /*pruneZeros*/ true);
  auto const tolerance = real(256 * kSize) * std::numeric_limits<real>::epsilon();
  BlockPrec<kBlockSize> prec(physical, /*fillInLevel*/ 0, /*alphaShift*/ 0_r);
  CheckMultiRhsMatchesColumnSolves<Matrix<real>>(prec, kSize, 123, tolerance);
  CheckMultiRhsMatchesColumnSolves<RowMatrix<real>>(prec, kSize, 124, tolerance);

  for (int i = 0; i < kSize; ++i) {
    physical.SetValue(i, i, dense(i, i) + 0.5_r);
  }
  prec.Update(physical);
  CheckMultiRhsMatchesColumnSolves<Matrix<real>>(prec, kSize, 125, tolerance);
  CheckMultiRhsMatchesColumnSolves<RowMatrix<real>>(prec, kSize, 126, tolerance);
}

TEST(IncompleteCholeskyPrec, BlockMultiRhs) {
  CheckBlockPrecMultiRhs<1>();
  CheckBlockPrecMultiRhs<2>();
  CheckBlockPrecMultiRhs<3>();
  CheckBlockPrecMultiRhs<4>();
}

TEST(IncompleteCholeskyPrec, BlockFillAndUpdateAreExact) {
  constexpr int kBlockSize = 4;
  constexpr int kBlockCount = 4;
  constexpr int kSize = kBlockSize * kBlockCount;
  constexpr real kTolerance = 8_r * kSize * std::numeric_limits<real>::epsilon();
  ColumnVector<real> x(kSize), rhs(kSize), result(kSize);
  x.SetRandom(123);

  for (int level = 1; level <= 2; ++level) {
    auto dense = test::Create2dLaplacianMatrix<real, kBlockSize>(2, 2);
    for (int block = 0; block < kBlockCount; ++block) {
      for (int row = 0; row < kBlockSize; ++row) {
        for (int col = 0; col < kBlockSize; ++col) {
          dense(block * kBlockSize + row, block * kBlockSize + col) += 0.05_r;
        }
      }
    }
    auto physical = ToBlockSparseMatrix<kBlockSize>(dense, /*pruneZeros*/ true);
    BlockPrec<kBlockSize> prec(physical, level, /*alphaShift*/ 0_r);
    auto checkExact = [&] {
      rhs = dense * x;
      prec(rhs, result);
      result -= x;
      EXPECT_LT(result.Norm(), x.Norm() * kTolerance);
    };

    checkExact();
    for (int i = 0; i < kSize; ++i) {
      dense(i, i) += 0.5_r;
      physical.SetValue(i, i, dense(i, i));
    }
    prec.Update(physical);
    checkExact();
  }
}

TEST(IC0B1SpMTest, Example1) {
  auto ABSp = ToBlockSparseMatrix<1>(A, /*pruneZeros*/ true);
  BlockPrec<1> p(ABSp, /*fillInLevel*/ 0, /*alphaShift*/ 0_r);
  Matrix<real> x(A.Rows(), 2), Px(A.Rows(), 2), R0Px(A.Rows(), 2);
  x.SetRandom(26);
  p(x, Px);
  R0Px = R0 * Px;
  Matrix<real> y = Transpose(R0) * R0Px;
  y -= x;
  auto const tol = std::numeric_limits<real>::epsilon() * A.Rows() * 2.0_r;
  EXPECT_LT(y.Norm(), x.Norm() * tol);

  RowMatrix<real> rowX(A.Rows(), 2), rowPx(A.Rows(), 2), rowR0Px(A.Rows(), 2);
  rowX.SetRandom(26);
  p(rowX, rowPx);
  rowR0Px = R0 * rowPx;
  RowMatrix<real> rowY = Transpose(R0) * rowR0Px;
  rowY -= rowX;
  EXPECT_LT(rowY.Norm(), rowX.Norm() * tol);

  auto MBSp = ToBlockSparseMatrix<1>(M, /*pruneZeros*/ true);
  BlockPrec<1> q(MBSp, /*fillInLevel*/ 0, /*alphaShift*/ 0.625_r);
  Matrix<real> shiftedX(M.Rows(), 2), shiftedPx(M.Rows(), 2), uPx(M.Rows(), 2);
  shiftedX.SetRandom(27);
  q(shiftedX, shiftedPx);
  uPx = U * shiftedPx;
  Matrix<real> shiftedY = Transpose(U) * uPx;
  shiftedY -= shiftedX;
  EXPECT_LT(shiftedY.Norm(), shiftedX.Norm() * tol);
}

template <typename MatrixType, int kBlockSize>
void CheckExactBlockPrec() {
  auto ABSp = ToBlockSparseMatrix<kBlockSize>(A, /*pruneZeros*/ true);
  BlockPrec<kBlockSize> prec(ABSp, /*fillInLevel*/ 0, /*alphaShift*/ 0_r);
  auto const tol = real(8 * A.Rows()) * std::numeric_limits<real>::epsilon();
  MatrixType X(A.Rows(), 3), Y(A.Rows(), 3), result(A.Rows(), 3);
  X.SetRandom(123);
  Y = A * X;
  prec(Y, result);
  result -= X;
  EXPECT_LT(result.Norm(), X.Norm() * tol);
}

TEST(IC0B2SpMTest, Example1) {
  CheckExactBlockPrec<Matrix<real>, 2>();
  CheckExactBlockPrec<RowMatrix<real>, 2>();
}

TEST(IC0B3SpMTest, Example1) {
  CheckExactBlockPrec<Matrix<real>, 3>();
  CheckExactBlockPrec<RowMatrix<real>, 3>();
}

class IC0MatAccess : public mochi::krylov::IncompleteCholeskyPrec<Matrix<real>> {
  using mochi::krylov::IncompleteCholeskyPrec<Matrix<real>>::IncompleteCholeskyPrec;
  friend class IC0MatTest;
  FRIEND_TEST(IC0MatTest, Example1);
};

class IC0MatTest : public testing::Test {
 protected:
  using PrecType = IC0MatAccess;
  void SetUp() override {
    _p = std::make_unique<PrecType>(A, /*fillInLevel*/ 0, /*alphaShift*/ 0_r);
    _q = std::make_unique<PrecType>(
        M, /*fillInLevel*/ 0, /*alphaShift*/ 0.625_r); // Shift yields the matrix M + I
  }
  std::unique_ptr<PrecType> _p = nullptr;
  std::unique_ptr<PrecType> _q = nullptr;
};

TEST_F(IC0MatTest, Example1) {
  int n = _p->_rIC.Rows();
  //--- Check symmetry
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      EXPECT_NEAR_EQ(_p->_rIC(i, j), _p->_rIC(j, i));
    }
  }
  //--- Check diagonal entries
  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR_EQ(_p->_rIC(i, i), 1.0_r / R0(i, i));
  }
  //--- Check off-diagonal entries
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < _p->_rIC.Cols(); ++j) {
      EXPECT_NEAR_EQ(_p->_rIC(i, j), R0(i, j));
    }
  }
  //--- Check operator()
  {
    Matrix<real> x(n, 3), Px(n, 3), R0Px(n, 3);
    x.SetRandom(22);
    _p->operator()(x, Px);
    R0Px = R0 * Px;
    Matrix<real> y = Transpose(R0) * R0Px;
    y -= x;
    auto tol = std::numeric_limits<real>::epsilon() * n;
    EXPECT_LT(y.Norm(), x.Norm() * tol);
  }
  {
    RowMatrix<real> x(n, 2), Px(n, 2), R0Px(n, 2);
    x.SetRandom(26);
    _p->operator()(x, Px); // Px = R0^{-1} * R0^{-T} * x
    R0Px = R0 * Px;
    RowMatrix<real> y = Transpose(R0) * R0Px;
    y -= x;
    auto tol = std::numeric_limits<real>::epsilon() * n * 2.0_r;
    EXPECT_LT(y.Norm(), x.Norm() * tol);
  }
  //
  n = M.Rows();
  //--- Check symmetry
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      EXPECT_NEAR_EQ(_q->_rIC(i, j), _q->_rIC(j, i));
    }
  }
  //--- Check diagonal entries
  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR_EQ(_q->_rIC(i, i), 1.0_r / U(i, i));
  }
  //--- Check off-diagonal entries
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < _q->_rIC.Cols(); ++j) {
      EXPECT_NEAR_EQ(_q->_rIC(i, j), U(i, j));
    }
  }
}

class ICSpMAccess
    : public mochi::krylov::IncompleteCholeskyPrec<SparseMatrix<real, int, int, std::vector>> {
  using mochi::krylov::IncompleteCholeskyPrec<
      SparseMatrix<real, int, int, std::vector>>::IncompleteCholeskyPrec;
  //
  friend class IC0SpMTest1;
  FRIEND_TEST(IC0SpMTest1, Example);
  //
  friend class IC1SpMTest1;
  FRIEND_TEST(IC1SpMTest1, Example);
  //
  friend class IC2SpMTest1;
  FRIEND_TEST(IC2SpMTest1, Example);
};

class IC0SpMTest1 : public testing::Test {
 protected:
  using PrecType = ICSpMAccess;
  void SetUp() override {
    auto Asp = ToSparseMatrix(A, /*pruneZeros*/ true);
    _p = std::make_unique<PrecType>(Asp, /*fillInLevel*/ 0, /*alphaShift*/ 0_r);
    //
    auto MSp = ToSparseMatrix(M, /*pruneZeros*/ true);
    _q = std::make_unique<PrecType>(
        MSp, /*fillInLevel*/ 0, /*alphaShift*/ 0.625_r); // Shift yields the matrix M + I
  }
  std::unique_ptr<PrecType> _p = nullptr;
  std::unique_ptr<PrecType> _q = nullptr;
};

TEST_F(IC0SpMTest1, Example) {
  int n = _p->_rChol.Rows();
  //--- Check symmetry
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      EXPECT_NEAR_EQ(_p->_rChol(i, j), _p->_rChol(j, i));
    }
  }
  //--- Check diagonal entries
  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR_EQ(_p->_rChol(i, i), 1.0_r / R0(i, i));
  }
  //--- Check off-diagonal entries
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < _p->_rChol.Cols(); ++j) {
      EXPECT_NEAR_EQ(_p->_rChol(i, j), R0(i, j));
    }
  }
  //--- Check operator()
  {
    Matrix<real> x(n, 3), Px(n, 3), R0Px(n, 3);
    x.SetRandom(26);
    _p->operator()(x, Px); // Px = R0^{-1} * R0^{-T} * x
    R0Px = R0 * Px;
    Matrix<real> y = Transpose(R0) * R0Px;
    y -= x;
    auto tol = std::numeric_limits<real>::epsilon() * real(n);
    EXPECT_LT(y.Norm(), x.Norm() * tol);
  }
  {
    RowMatrix<real> x(n, 2), Px(n, 2), R0Px(n, 2);
    x.SetRandom(26);
    _p->operator()(x, Px); // Px = R0^{-1} * R0^{-T} * x
    R0Px = R0 * Px;
    RowMatrix<real> y = Transpose(R0) * R0Px;
    y -= x;
    auto tol = std::numeric_limits<real>::epsilon() * real(n) * 2.0_r;
    EXPECT_LT(y.Norm(), x.Norm() * tol);
  }
  //
  n = M.Rows();
  //--- Check symmetry
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      EXPECT_NEAR_EQ(_q->_rChol(i, j), _q->_rChol(j, i));
    }
  }
  //--- Check diagonal entries
  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR_EQ(_q->_rChol(i, i), 1.0_r / U(i, i));
  }
  //--- Check off-diagonal entries
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < _q->_rChol.Cols(); ++j) {
      EXPECT_NEAR_EQ(_q->_rChol(i, j), U(i, j));
    }
  }
}

static DynamicArray<int> hRowPtr({0, 4, 6, 9, 13, 16, 20});
static DynamicArray<int> hColIdx({0, 2, 3, 5, 1, 3, 0, 2, 4, 0, 1, 3, 5, 2, 4, 5, 0, 3, 4, 5});
static DynamicArray<real> hValues({3,  -1, -1, -1, 2, -1, -1, 3,  -1, -1,
                                   -1, 2,  -1, -1, 3, -1, -1, -1, -1, 4});
static SparseMatrix<real> H(6, std::move(hRowPtr), std::move(hColIdx), std::move(hValues));

class IC1SpMTest1 : public testing::Test {
 protected:
  using PrecType = ICSpMAccess;
  void SetUp() override {
    _p = std::make_unique<PrecType>(H, /*fillInLevel*/ 1, /*alphaShift*/ 0_r);
  }
  std::unique_ptr<PrecType> _p = nullptr;
};

TEST_F(IC1SpMTest1, Example) {
  //
  // This example came from a presentation of K. Meerbergen
  // https://people.cs.kuleuven.be/~karl.meerbergen/didactiek/h03g1a/ilu.pdf
  //
  Matrix<real, 6, 6> HR(
      1_r / Sqrt(3_r),
      0_r,
      -1_r / Sqrt(3_r),
      -1_r / Sqrt(3_r),
      0_r,
      -1_r / Sqrt(3_r),
      0_r,
      1_r / Sqrt(2_r),
      0_r,
      -1_r / Sqrt(2_r),
      0_r,
      0_r,
      -1_r / Sqrt(3_r),
      0_r,
      0.612372_r,
      -0.204124_r,
      -0.612372_r,
      -0.204124_r,
      -1_r / Sqrt(3_r),
      -1_r / Sqrt(2_r),
      -0.204124_r,
      0.942809_r,
      -0.117851_r,
      -1.29636_r,
      0_r,
      0_r,
      -0.612372_r,
      -0.117851_r,
      0.618853_r,
      -0.790756_r,
      -1_r / Sqrt(3_r),
      0_r,
      -0.204124_r,
      -1.29636_r,
      -0.790756_r,
      0.870669_r);
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      EXPECT_NEAR_RTOL(_p->_rChol(i, j), HR(i, j), 0.00001_r);
    }
  }
}

class IC2SpMTest1 : public testing::Test {
 protected:
  using PrecType = ICSpMAccess;
  void SetUp() override {
    //--- Level fill-in 2 has the sparsity for the full factorization
    _p = std::make_unique<PrecType>(H, /*fillInLevel*/ 2, /*alphaShift*/ 0_r);
  }
  std::unique_ptr<PrecType> _p = nullptr;
};

TEST_F(IC2SpMTest1, Example) {
  //--- Here the factorization is exact
  auto tol = real(4 * H.Rows()) * std::numeric_limits<real>::epsilon();
  {
    Matrix<real> X(H.Rows(), 6);
    X.SetRandom(123);
    Matrix<real> Y(X), PY(X);
    Y = H * X;
    _p->operator()(Y, PY);
    //--- PY should match X
    PY -= X;
    EXPECT_LT(PY.Norm(), X.Norm() * tol);
  }
  {
    RowMatrix<real> X(H.Rows(), 6);
    X.SetRandom(456);
    RowMatrix<real> Y(X), PY(X);
    Y = H * X;
    _p->operator()(Y, PY);
    //--- PY should match X
    PY -= X;
    EXPECT_LT(PY.Norm(), X.Norm() * tol);
  }
}

TEST(IncompleteCholeskyPrec, Update) {
  // Test that Update produces the same result as creating a new preconditioner.

  for (int level = 0; level <= 2; ++level) {
    // Test with SparseMatrix
    {
      auto A1Sp = ToSparseMatrix(A, /*pruneZeros*/ true);
      auto A2 = RowMatrix<real, 6, 6>(A);
      A2 *= 1.2_r; // Scale to get different values
      auto A2Sp = ToSparseMatrix(A2, /*pruneZeros*/ true);

      krylov::IncompleteCholeskyPrec<SparseMatrix<real>> P1(
          A1Sp, level, /*alphaShift*/ Scalar(0.1));
      krylov::IncompleteCholeskyPrec<SparseMatrix<real>> P2(
          A2Sp, level, /*alphaShift*/ Scalar(0.1));

      P1.Update(A2Sp);

      ColumnVector<real> x(6), Px1(6), Px2(6);
      x.SetRandom(123);
      P1(x, Px1);
      P2(x, Px2);

      auto tol = real(8 * 6) * std::numeric_limits<real>::epsilon();
      EXPECT_TRUE(mochi::test::NearEqualMatrices(Px1, Px2, tol));
    }

    // Test with BlockSparseMatrix
    {
      auto A1BSp = ToBlockSparseMatrix<2>(A, /*pruneZeros*/ true);
      auto A2 = RowMatrix<real, 6, 6>(A);
      A2 *= 1.3_r;
      auto A2BSp = ToBlockSparseMatrix<2>(A2, /*pruneZeros*/ true);

      krylov::IncompleteCholeskyPrec<BlockSparseMatrix<real, 2>> P1(
          A1BSp, level, /*alphaShift*/ Scalar(0.1));
      krylov::IncompleteCholeskyPrec<BlockSparseMatrix<real, 2>> P2(
          A2BSp, level, /*alphaShift*/ Scalar(0.1));

      P1.Update(A2BSp);

      ColumnVector<real> x(6), Px1(6), Px2(6);
      x.SetRandom(456);
      P1(x, Px1);
      P2(x, Px2);

      auto tol = real(8 * 6) * std::numeric_limits<real>::epsilon();
      EXPECT_TRUE(mochi::test::NearEqualMatrices(Px1, Px2, tol));
    }

    // Test with dense Matrix
    if (level == 0) { // Only supported without fill-in
      Matrix<real> A1(A);
      Matrix<real> A2(A);
      A2 *= 1.4_r;

      krylov::IncompleteCholeskyPrec<Matrix<real>> P1(A1, level, /*alphaShift*/ Scalar(0.1));
      krylov::IncompleteCholeskyPrec<Matrix<real>> P2(A2, level, /*alphaShift*/ Scalar(0.1));

      P1.Update(A2);

      ColumnVector<real> x(6), Px1(6), Px2(6);
      x.SetRandom(789);
      P1(x, Px1);
      P2(x, Px2);

      auto tol = real(8 * 6) * std::numeric_limits<real>::epsilon();
      EXPECT_TRUE(mochi::test::NearEqualMatrices(Px1, Px2, tol));
    }
  }
}

/// @brief Test ConcurrentSolve gives same result as Solve for IC0.
/// @todo Re-enable when implementing ConcurrentSolve and extend to actual multi-threaded tests.
TEST(IncompleteCholeskyPrec, DISABLED_ConcurrentSolve) {
  // Test with SparseMatrix
  {
    auto Asp = ToSparseMatrix(A, /*pruneZeros*/ true);

    krylov::IncompleteCholeskyPrec<SparseMatrix<real>> P(Asp, 0, /*alphaShift*/ Scalar(0));

    ColumnVector<real> x(6), Px1(6), Px2(6);
    x.SetRandom(123);
    Px1.SetRandom(456);
    Px2.SetRandom(789);

    P.Solve(x, Px1);
    ParallelBarrier barrier(1);
    P.ConcurrentSolve(x, Px2, {0, 1, 0, 6, barrier});

    EXPECT_TRUE(mochi::test::NearEqualMatrices(Px1, Px2));
  }

  // Test with BlockSparseMatrix
  {
    auto ABSp = ToBlockSparseMatrix<2>(A, /*pruneZeros*/ true);

    krylov::IncompleteCholeskyPrec<BlockSparseMatrix<real, 2>> P(ABSp, 0, /*alphaShift*/ Scalar(0));

    ColumnVector<real> x(6), Px1(6), Px2(6);
    x.SetRandom(123);
    Px1.SetRandom(456);
    Px2.SetRandom(789);

    P.Solve(x, Px1);
    ParallelBarrier barrier(1);
    P.ConcurrentSolve(x, Px2, {0, 1, 0, 6, barrier});

    EXPECT_TRUE(mochi::test::NearEqualMatrices(Px1, Px2));
  }

  // Test with dense Matrix
  {
    Matrix<real> Am(A);

    krylov::IncompleteCholeskyPrec<Matrix<real>> P(Am, 0, /*alphaShift*/ Scalar(0));

    ColumnVector<real> x(6), Px1(6), Px2(6);
    x.SetRandom(456);
    Px1.SetRandom(789);
    Px2.SetRandom(111);

    P.Solve(x, Px1);
    ParallelBarrier barrier(1);
    P.ConcurrentSolve(x, Px2, {0, 1, 0, 6, barrier});

    EXPECT_TRUE(mochi::test::NearEqualMatrices(Px1, Px2));
  }
}
