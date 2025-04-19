#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/Sadikov_I_SparseMatrixMultiplication_OMP/include/SparseMatrix.hpp"
#include "omp/Sadikov_I_SparseMatrixMultiplication_OMP/include/ops_omp.hpp"

namespace {
std::vector<double> GetRandomMatrix(int size) {
  std::vector<double> data(size);
  std::random_device dev;
  std::mt19937 gen(dev());
  int high = 500;
  int low = 0;
  std::uniform_int_distribution<> number(low, high);
  for (int i = 0; i < size / 5; ++i) {
    data[i] = static_cast<double>(number(gen));
  }
  std::ranges::shuffle(data, gen);
  return data;
}
}  // namespace

TEST(sadikov_i_sparse_matrix_multiplication_task_omp, test_rect_matrixes) {
  constexpr auto kEpsilon = 0.000001;
  std::vector<double> fmatrix{0, 0, 0, 5.0, 2.0, 0, 1.0, 0, 7.0, 7.0, 0, 0};
  std::vector<double> smatrix{1.0, 0, 0, 2.0, 0, 8.0, 0, 0, 0, 0, 5.0, 0};
  std::vector<double> out(9, 0.0);
  std::vector<double> test_out{0.0, 25.0, 0.0, 2.0, 0.0, 0.0, 21.0, 0.0, 56.0};
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(fmatrix.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(smatrix.data()));
  task_data_seq->inputs_count.emplace_back(3);
  task_data_seq->inputs_count.emplace_back(4);
  task_data_seq->inputs_count.emplace_back(4);
  task_data_seq->inputs_count.emplace_back(3);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  sadikov_i_sparse_matrix_multiplication_task_omp::CCSMatrixOMP test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (auto i = 0; i < static_cast<int>(out.size()); ++i) {
    EXPECT_NEAR(out[i], test_out[i], kEpsilon);
  }
}

TEST(sadikov_i_sparse_matrix_multiplication_task_omp, test_square_matrixes) {
  constexpr auto kEpsilon = 0.000001;
  std::vector<double> fmatrix{1.0, 0.0, 0.0, 0.0, 7.0, 0.0, 4.0, 9.0, 0.0};
  std::vector<double> smatrix{0.0, 0.0, 3.0, 2.0, 0.0, 0.0, 10.0, 0.0, 0.0};
  std::vector<double> out(9, 0.0);
  std::vector<double> test_out{0.0, 0.0, 3.0, 14.0, 0.0, 0.0, 18.0, 0.0, 12.0};
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(fmatrix.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(smatrix.data()));
  for (auto i = 0; i < 4; ++i) {
    task_data_seq->inputs_count.emplace_back(3);
  }
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  sadikov_i_sparse_matrix_multiplication_task_omp::CCSMatrixOMP test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (auto i = 0; i < static_cast<int>(out.size()); ++i) {
    EXPECT_NEAR(out[i], test_out[i], kEpsilon);
  }
}

TEST(sadikov_i_sparse_matrix_multiplication_task_omp, test_empty_matrixes) {
  constexpr auto kEpsilon = 0.000001;
  std::vector<double> fmatrix;
  std::vector<double> smatrix;
  std::vector<double> out;
  std::vector<double> test_out;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(fmatrix.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(smatrix.data()));
  task_data_seq->inputs_count.emplace_back(0);
  task_data_seq->inputs_count.emplace_back(0);
  task_data_seq->inputs_count.emplace_back(0);
  task_data_seq->inputs_count.emplace_back(0);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  sadikov_i_sparse_matrix_multiplication_task_omp::CCSMatrixOMP test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (auto i = 0; i < static_cast<int>(out.size()); ++i) {
    EXPECT_NEAR(out[i], test_out[i], kEpsilon);
  }
}

TEST(sadikov_i_sparse_matrix_multiplication_task_omp, test_random_matrix) {
  constexpr auto kEpsilon = 0.000001;
  constexpr auto kSize = 40;
  auto fmatrix = GetRandomMatrix(kSize * kSize);
  auto smatrix = GetRandomMatrix(kSize * kSize);
  std::vector<double> out(kSize * kSize, 0.0);
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(fmatrix.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(smatrix.data()));
  task_data_seq->inputs_count.emplace_back(kSize);
  task_data_seq->inputs_count.emplace_back(kSize);
  task_data_seq->inputs_count.emplace_back(kSize);
  task_data_seq->inputs_count.emplace_back(kSize);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  auto check_out = sadikov_i_sparse_matrix_multiplication_task_omp::BaseMatrixMultiplication(fmatrix, kSize, kSize,
                                                                                             smatrix, kSize, kSize);
  sadikov_i_sparse_matrix_multiplication_task_omp::CCSMatrixOMP test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (auto i = 0; i < static_cast<int>(out.size()); ++i) {
    EXPECT_NEAR(out[i], check_out[i], kEpsilon);
  }
}