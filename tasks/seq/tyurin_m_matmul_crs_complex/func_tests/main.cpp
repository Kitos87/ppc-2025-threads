#include <gtest/gtest.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/tyurin_m_matmul_crs_complex/include/ops_seq.hpp"

namespace {
Matrix RandMatrix(uint32_t rows, uint32_t cols, double percentage) {
  std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<double> distr(-10000, 10000);
  Matrix res{.rows = rows, .cols = cols, .data = std::vector<std::complex<double>>(rows * cols)};
  std::ranges::generate(res.data, [&]() {
    const auto el = distr(gen);
    const auto re = (el < (distr.min() + ((distr.max() - distr.min()) * percentage))) ? el : 0;

    std::complex<double> cmplx;
    cmplx.real(re);
    if (re != 0.0) {
      cmplx.imag(distr(gen));
    }

    return cmplx;
  });
  return res;
}
void TestMatrixCRS(Matrix &&lhs, Matrix &&rhs) {
  MatrixCRS crs_lhs = RegularToCRS(lhs);
  MatrixCRS crs_rhs = RegularToCRS(rhs);
  MatrixCRS crs_out;

  auto data = std::make_shared<ppc::core::TaskData>();
  data->inputs = {reinterpret_cast<uint8_t *>(&crs_lhs), reinterpret_cast<uint8_t *>(&crs_rhs)};
  data->inputs_count = {lhs.rows, lhs.cols, rhs.rows, rhs.cols};
  data->outputs = {reinterpret_cast<uint8_t *>(&crs_out)};
  data->outputs_count = {1};

  tyurin_m_matmul_crs_complex_seq::TestTaskSequential task(data);
  ASSERT_EQ(task.Validation(), true);
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  Matrix regular_out = CRSToRegular(crs_out);
  EXPECT_EQ(regular_out, MultiplyMat(lhs, rhs));
}
}  // namespace

// clang-format off
TEST(tyurin_m_matmul_crs_complex_seq, test_regular_matrix_mult_1) {
  Matrix lhs{ .rows=5, .cols=5, .data={
    43, 46, 21, 21, 87,
    39, 26, 82, 65, 62,
    97, 47, 32, 16, 61,
    76, 43, 78, 50, 63,
    18, 14, 84, 22, 55
  }};
  Matrix rhs{ .rows=5, .cols=5, .data={
    43, 46, 21, 21, 87,
    39, 26, 82, 65, 62,
    97, 47, 32, 16, 61,
    76, 43, 78, 50, 63,
    18, 14, 84, 22, 55
  }};
  Matrix ref{ .rows=5, .cols=5, .data={
    8842, 6282, 14293, 7193, 13982,
    16701, 9987, 15853, 8435, 17512,
    11422, 8730, 13287, 7746, 17668,
    17445, 11312, 16810, 9525, 20651,
    12130, 6856, 10550, 4942, 11969
  }};
  EXPECT_EQ(MultiplyMat(lhs, rhs), ref);
}
TEST(tyurin_m_matmul_crs_complex_seq, test_regular_matrix_mult_2) {
  Matrix lhs{ .rows=5, .cols=4, .data={
    43, 46, 21, 21,
    39, 26, 82, 65,
    97, 47, 32, 16,
    76, 43, 78, 50,
    18, 14, 84, 22
  }};
  Matrix rhs{ .rows=4, .cols=5, .data={
    43, 46, 21, 21, 87,
    39, 26, 82, 65, 62,
    97, 47, 32, 16, 61,
    76, 43, 78, 50, 63
  }};
  Matrix ref{ .rows=5, .cols=5, .data={
    7276, 5064, 6985, 5279, 9197,
    15585, 9119, 10645, 7071, 14102,
    10324, 7876, 8163, 6404, 14313,
    16311, 10430, 11518, 8139, 17186,
    11140, 6086, 5930, 3732, 8944
  }};
  EXPECT_EQ(MultiplyMat(lhs, rhs), ref);
}
// clang-format on

TEST(tyurin_m_matmul_crs_complex_seq, test_crs_random_30x30p00mul30x30p00) {
  TestMatrixCRS(RandMatrix(30, 30, .0), RandMatrix(30, 30, .0));
}
TEST(tyurin_m_matmul_crs_complex_seq, test_crs_random_30x30p20mul30x30p20) {
  TestMatrixCRS(RandMatrix(30, 30, .20), RandMatrix(30, 30, .20));
}
TEST(tyurin_m_matmul_crs_complex_seq, test_crs_random_30x30p20mul30x30p50) {
  TestMatrixCRS(RandMatrix(30, 30, .20), RandMatrix(30, 30, .50));
}
TEST(tyurin_m_matmul_crs_complex_seq, test_crs_random_30x30p70mul30x30p50) {
  TestMatrixCRS(RandMatrix(30, 30, .70), RandMatrix(30, 30, .50));
}
TEST(tyurin_m_matmul_crs_complex_seq, test_crs_random_30x30p70mul30x30p20) {
  TestMatrixCRS(RandMatrix(30, 30, .70), RandMatrix(30, 30, .20));
}
TEST(tyurin_m_matmul_crs_complex_seq, test_crs_random_30x40p70mul40x30p60) {
  TestMatrixCRS(RandMatrix(30, 40, .70), RandMatrix(40, 30, .60));
}
TEST(tyurin_m_matmul_crs_complex_seq, test_crs_random_30x23p70mul23x30p63) {
  TestMatrixCRS(RandMatrix(30, 23, .70), RandMatrix(23, 30, .63));
}
TEST(tyurin_m_matmul_crs_complex_seq, test_crs_random_30x1p70mul1x1p63) {
  TestMatrixCRS(RandMatrix(30, 1, .70), RandMatrix(1, 30, .63));
}
TEST(tyurin_m_matmul_crs_complex_seq, test_crs_random_30x1p38mul1x1p63) {
  TestMatrixCRS(RandMatrix(30, 1, .38), RandMatrix(1, 30, .63));
}
TEST(tyurin_m_matmul_crs_complex_seq, test_regular_matrix_mult_inv) {
  Matrix lhs{.rows = 3, .cols = 3, .data = {1, 0, 0, 1, -1, 0, 1, 0, 1}};
  Matrix rhs{.rows = 3, .cols = 3, .data = {1, 0, 0, 1, -1, 0, -1, 0, 1}};
  Matrix ref{.rows = 3, .cols = 3, .data = {1, 0, 0, 0, 1, 0, 0, 0, 1}};
  EXPECT_EQ(MultiplyMat(lhs, rhs), ref);
}
TEST(tyurin_m_matmul_crs_complex_seq, test_crs_random_inv) {
  TestMatrixCRS({.rows = 3, .cols = 3, .data = {1, 0, 0, 1, -1, 0, 1, 0, 1}},
                {.rows = 3, .cols = 3, .data = {1, 0, 0, 1, -1, 0, -1, 0, 1}});
}
TEST(tyurin_m_matmul_crs_complex_seq, validation_failure) {
  const auto lhs = RandMatrix(30, 40, .70);
  const auto rhs = RandMatrix(50, 50, .70);

  MatrixCRS crs_lhs = RegularToCRS(lhs);
  MatrixCRS crs_rhs = RegularToCRS(rhs);
  MatrixCRS crs_out;

  auto data = std::make_shared<ppc::core::TaskData>();
  data->inputs = {reinterpret_cast<uint8_t *>(&crs_lhs), reinterpret_cast<uint8_t *>(&crs_rhs)};
  data->inputs_count = {lhs.rows, lhs.cols, rhs.rows, rhs.cols};
  data->outputs = {reinterpret_cast<uint8_t *>(&crs_out)};
  data->outputs_count = {1};

  tyurin_m_matmul_crs_complex_seq::TestTaskSequential task(data);
  EXPECT_FALSE(task.Validation());
}
