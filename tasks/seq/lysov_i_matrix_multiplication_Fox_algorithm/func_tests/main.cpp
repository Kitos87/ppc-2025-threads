#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/lysov_i_matrix_multiplication_Fox_algorithm/include/ops_seq.hpp"

namespace lysov_i_matrix_multiplication_fox_algorithm_seq {
std::vector<double> GetRandomMatrix(size_t size, int min_gen_value, int max_gen_value) {
  std::vector<double> matrix(size * size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(min_gen_value, max_gen_value);

  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      matrix[(i * size) + j] = dist(gen);
    }
  }
  return matrix;
}
void TrivialMatrixMultiplication(const std::vector<double> &matrix_a, const std::vector<double> &matrix_b,
                                 std::vector<double> &result_matrix, size_t matrix_size) {
  for (size_t row = 0; row < matrix_size; ++row) {
    for (size_t col = 0; col < matrix_size; ++col) {
      result_matrix[(row * matrix_size) + col] = 0.0;
      for (size_t k = 0; k < matrix_size; ++k) {
        result_matrix[(row * matrix_size) + col] +=
            matrix_a[(row * matrix_size) + k] * matrix_b[(k * matrix_size) + col];
      }
      result_matrix[(row * matrix_size) + col] = round(result_matrix[(row * matrix_size) + col] * 10000) / 10000;
    }
  }
}
}  // namespace lysov_i_matrix_multiplication_fox_algorithm_seq

TEST(lysov_i_matrix_multiplication_fox_algorithm_seq, Test_Matrix_Multiplication_Identity) {
  size_t n = 3;
  size_t block_size = 2;
  std::vector<double> a = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> b = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<double> c(n * n, 0);
  std::shared_ptr<ppc::core::TaskData> task_data_sequential = std::make_shared<ppc::core::TaskData>();
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block_size));
  task_data_sequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data_sequential->inputs_count.emplace_back(n * n);
  task_data_sequential->inputs_count.emplace_back(n * n);
  task_data_sequential->inputs_count.emplace_back(1);
  task_data_sequential->outputs_count.emplace_back(n * n);
  lysov_i_matrix_multiplication_fox_algorithm_seq::TestTaskSequential matrix_multiplication(task_data_sequential);
  ASSERT_EQ(matrix_multiplication.Validation(), true);
  matrix_multiplication.PreProcessing();
  matrix_multiplication.Run();
  matrix_multiplication.PostProcessing();
  EXPECT_EQ(c, a);
}

TEST(lysov_i_matrix_multiplication_fox_algorithm_seq, Test_Matrix_Multiplication_Arbitrary_Values) {
  size_t n = 3;
  size_t block_size = 1;
  std::vector<double> a = {2, 3, 1, 4, 0, 5, 1, 2, 3};
  std::vector<double> b = {1, 2, 3, 0, 1, 0, 4, 0, 1};
  std::vector<double> c(n * n, 0);
  std::vector<double> c_expected = {6.0, 7.0, 7.0, 24.0, 8.0, 17.0, 13.0, 4.0, 6.0};
  std::shared_ptr<ppc::core::TaskData> task_data_sequential = std::make_shared<ppc::core::TaskData>();
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block_size));
  task_data_sequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data_sequential->inputs_count.emplace_back(n * n);
  task_data_sequential->inputs_count.emplace_back(n * n);
  task_data_sequential->inputs_count.emplace_back(1);
  task_data_sequential->outputs_count.emplace_back(n * n);
  lysov_i_matrix_multiplication_fox_algorithm_seq::TestTaskSequential matrix_multiplication(task_data_sequential);
  ASSERT_EQ(matrix_multiplication.ValidationImpl(), true);
  matrix_multiplication.PreProcessingImpl();
  matrix_multiplication.RunImpl();
  matrix_multiplication.PostProcessingImpl();
  EXPECT_EQ(c, c_expected);
}

TEST(lysov_i_matrix_multiplication_fox_algorithm_seq, Test_matrix_7x7) {
  size_t n = 7;
  size_t block_size = 3;
  int min_gen_value = -1e3;
  int max_gen_value = 1e3;
  std::vector<double> a =
      lysov_i_matrix_multiplication_fox_algorithm_seq::GetRandomMatrix(n, min_gen_value, max_gen_value);
  std::vector<double> b =
      lysov_i_matrix_multiplication_fox_algorithm_seq::GetRandomMatrix(n, min_gen_value, max_gen_value);
  std::vector<double> c(n * n, 0);
  std::vector<double> c_expected(n * n, 0);
  lysov_i_matrix_multiplication_fox_algorithm_seq::TrivialMatrixMultiplication(a, b, c_expected, n);
  std::shared_ptr<ppc::core::TaskData> task_data_sequential = std::make_shared<ppc::core::TaskData>();
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block_size));
  task_data_sequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data_sequential->inputs_count.emplace_back(n * n);
  task_data_sequential->inputs_count.emplace_back(n * n);
  task_data_sequential->inputs_count.emplace_back(1);
  task_data_sequential->outputs_count.emplace_back(n * n);
  lysov_i_matrix_multiplication_fox_algorithm_seq::TestTaskSequential matrix_multiplication(task_data_sequential);
  ASSERT_EQ(matrix_multiplication.ValidationImpl(), true);
  matrix_multiplication.PreProcessingImpl();
  matrix_multiplication.RunImpl();
  matrix_multiplication.PostProcessingImpl();
  for (size_t i = 0; i < n * n; i++) {
    EXPECT_NEAR(c_expected[i], c[i], 1e-3);
  }
}

TEST(lysov_i_matrix_multiplication_fox_algorithm_seq, Test_matrix_16x16) {
  size_t n = 16;
  size_t block_size = 4;
  int min_gen_value = -1e3;
  int max_gen_value = 1e3;
  std::vector<double> a =
      lysov_i_matrix_multiplication_fox_algorithm_seq::GetRandomMatrix(n, min_gen_value, max_gen_value);
  std::vector<double> b =
      lysov_i_matrix_multiplication_fox_algorithm_seq::GetRandomMatrix(n, min_gen_value, max_gen_value);
  std::vector<double> c(n * n, 0);
  std::vector<double> c_expected(n * n, 0);
  lysov_i_matrix_multiplication_fox_algorithm_seq::TrivialMatrixMultiplication(a, b, c_expected, n);
  std::shared_ptr<ppc::core::TaskData> task_data_sequential = std::make_shared<ppc::core::TaskData>();
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block_size));
  task_data_sequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data_sequential->inputs_count.emplace_back(n * n);
  task_data_sequential->inputs_count.emplace_back(n * n);
  task_data_sequential->inputs_count.emplace_back(1);
  task_data_sequential->outputs_count.emplace_back(n * n);
  lysov_i_matrix_multiplication_fox_algorithm_seq::TestTaskSequential matrix_multiplication(task_data_sequential);
  ASSERT_EQ(matrix_multiplication.ValidationImpl(), true);
  matrix_multiplication.PreProcessingImpl();
  matrix_multiplication.RunImpl();
  matrix_multiplication.PostProcessingImpl();
  for (size_t i = 0; i < n * n; i++) {
    EXPECT_NEAR(c_expected[i], c[i], 1e-3);
  }
}

TEST(lysov_i_matrix_multiplication_fox_algorithm_seq, Test_matrix_11x11) {
  size_t n = 11;
  size_t block_size = 10;
  int min_gen_value = -1e3;
  int max_gen_value = 1e3;
  std::vector<double> a =
      lysov_i_matrix_multiplication_fox_algorithm_seq::GetRandomMatrix(n, min_gen_value, max_gen_value);
  std::vector<double> b =
      lysov_i_matrix_multiplication_fox_algorithm_seq::GetRandomMatrix(n, min_gen_value, max_gen_value);
  std::vector<double> c(n * n, 0);
  std::vector<double> c_expected(n * n, 0);
  lysov_i_matrix_multiplication_fox_algorithm_seq::TrivialMatrixMultiplication(a, b, c_expected, n);
  std::shared_ptr<ppc::core::TaskData> task_data_sequential = std::make_shared<ppc::core::TaskData>();
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block_size));
  task_data_sequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data_sequential->inputs_count.emplace_back(n * n);
  task_data_sequential->inputs_count.emplace_back(n * n);
  task_data_sequential->inputs_count.emplace_back(1);
  task_data_sequential->outputs_count.emplace_back(n * n);
  lysov_i_matrix_multiplication_fox_algorithm_seq::TestTaskSequential matrix_multiplication(task_data_sequential);
  ASSERT_EQ(matrix_multiplication.ValidationImpl(), true);
  matrix_multiplication.PreProcessingImpl();
  matrix_multiplication.RunImpl();
  matrix_multiplication.PostProcessingImpl();
  for (size_t i = 0; i < n * n; i++) {
    EXPECT_NEAR(c_expected[i], c[i], 1e-3);
  }
}

TEST(lysov_i_matrix_multiplication_fox_algorithm_seq, Test_matrix_1x1) {
  size_t n = 1;
  size_t block_size = 3;
  std::vector<double> a = {2};
  std::vector<double> b = {2};
  std::vector<double> c(n * n, 0);
  std::vector<double> c_expected(n * n, 4);
  std::shared_ptr<ppc::core::TaskData> task_data_sequential = std::make_shared<ppc::core::TaskData>();
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block_size));
  task_data_sequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data_sequential->inputs_count.emplace_back(n * n);
  task_data_sequential->inputs_count.emplace_back(n * n);
  task_data_sequential->inputs_count.emplace_back(1);
  task_data_sequential->outputs_count.emplace_back(n * n);
  lysov_i_matrix_multiplication_fox_algorithm_seq::TestTaskSequential matrix_multiplication(task_data_sequential);
  ASSERT_EQ(matrix_multiplication.ValidationImpl(), true);
  matrix_multiplication.PreProcessingImpl();
  matrix_multiplication.RunImpl();
  matrix_multiplication.PostProcessingImpl();
  EXPECT_EQ(c, c_expected);
}

TEST(lysov_i_matrix_multiplication_fox_algorithm_seq, Test_Matrix_Multiplication_Empty_MatrixA) {
  size_t n = 3;
  size_t block_size = 1;
  std::vector<double> a(0, 0);
  std::vector<double> b = {2, 98, 7, 6, 5, 4, 5, 6, 6};
  std::vector<double> c(n * n, 0);
  std::shared_ptr<ppc::core::TaskData> task_data_sequential = std::make_shared<ppc::core::TaskData>();
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block_size));
  task_data_sequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data_sequential->inputs_count.emplace_back(a.size());
  task_data_sequential->inputs_count.emplace_back(n * n);
  task_data_sequential->inputs_count.emplace_back(1);
  task_data_sequential->outputs_count.emplace_back(n * n);
  lysov_i_matrix_multiplication_fox_algorithm_seq::TestTaskSequential matrix_multiplication(task_data_sequential);
  ASSERT_FALSE(matrix_multiplication.ValidationImpl());
}

TEST(lysov_i_matrix_multiplication_fox_algorithm_seq, Test_Matrix_Multiplication_Empty_MatrixB) {
  size_t n = 3;
  size_t block_size = 1;
  std::vector<double> a = {2, 98, 7, 6, 5, 4, 5, 6, 6};
  std::vector<double> b(0, 0);
  std::vector<double> c(n * n, 0);
  std::shared_ptr<ppc::core::TaskData> task_data_sequential = std::make_shared<ppc::core::TaskData>();
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block_size));
  task_data_sequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data_sequential->inputs_count.emplace_back(n * n);
  task_data_sequential->inputs_count.emplace_back(b.size());
  task_data_sequential->inputs_count.emplace_back(1);
  task_data_sequential->outputs_count.emplace_back(n * n);
  lysov_i_matrix_multiplication_fox_algorithm_seq::TestTaskSequential matrix_multiplication(task_data_sequential);
  ASSERT_FALSE(matrix_multiplication.ValidationImpl());
}

TEST(lysov_i_matrix_multiplication_fox_algorithm_seq, Test_Matrix_Multiplication_Negative_Values) {
  size_t n = 3;
  size_t block_size = 2;
  std::vector<double> a = {-1, -3, -1, -4, 0, -5, -1, -2, -3};
  std::vector<double> b = {-1, -2, -3, 0, -1, 0, -4, 0, -1};
  std::vector<double> c(n * n, 0);
  std::vector<double> c_expected(n * n, 0);
  lysov_i_matrix_multiplication_fox_algorithm_seq::TrivialMatrixMultiplication(a, b, c_expected, n);
  std::shared_ptr<ppc::core::TaskData> task_data_sequential = std::make_shared<ppc::core::TaskData>();
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block_size));
  task_data_sequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data_sequential->inputs_count.emplace_back(n * n);
  task_data_sequential->inputs_count.emplace_back(n * n);
  task_data_sequential->inputs_count.emplace_back(1);
  task_data_sequential->outputs_count.emplace_back(n * n);
  lysov_i_matrix_multiplication_fox_algorithm_seq::TestTaskSequential matrix_multiplication(task_data_sequential);
  ASSERT_EQ(matrix_multiplication.ValidationImpl(), true);
  matrix_multiplication.PreProcessingImpl();
  matrix_multiplication.RunImpl();
  matrix_multiplication.PostProcessingImpl();
  for (size_t i = 0; i < n * n; i++) {
    EXPECT_NEAR(c_expected[i], c[i], 1e-3);
  }
}