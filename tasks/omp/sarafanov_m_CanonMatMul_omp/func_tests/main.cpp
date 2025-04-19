#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/sarafanov_m_CanonMatMul_omp/include/ops_omp.hpp"

namespace {
std::vector<double> GenerateRandomData(int size) {
  std::vector<double> matrix(size);
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> dist(-500, 10000);
  for (auto i = 0; i < size; ++i) {
    matrix[i] = static_cast<double>(dist(gen));
  }
  return matrix;
}

std::vector<double> GenerateSingleMatrix(int size) {
  std::vector<double> matrix(size, 0.0);
  int sqrt_size = static_cast<int>(std::sqrt(size));
  for (int i = 0; i < sqrt_size; ++i) {
    for (int j = 0; j < sqrt_size; ++j) {
      if (i == j) {
        matrix[(sqrt_size * i) + j] = 1.0;
      }
    }
  }
  return matrix;
}
}  // namespace

TEST(sarafanov_m_canon_mat_mul_seq, test_clear_matrix) {
  constexpr size_t kCount = 0;
  constexpr double kInaccuracy = 0.001;
  std::vector<double> a_matrix;
  std::vector<double> b_matrix;
  std::vector<double> test_data;
  std::vector<double> out(kCount, 0);
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_matrix.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_matrix.data()));
  for (int i = 0; i < 4; ++i) {
    task_data_omp->inputs_count.emplace_back(kCount);
  }
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(kCount);

  sarafanov_m_canon_mat_mul_omp::CanonMatMulOMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  for (size_t i = 0; i < kCount; ++i) {
    EXPECT_NEAR(out[i], test_data[i], kInaccuracy);
  }
}

TEST(sarafanov_m_canon_mat_mul_omp, test_1x1_matrix) {
  constexpr size_t kCount = 1;
  constexpr double kInaccuracy = 0.001;
  std::vector<double> a_matrix{18.0};
  std::vector<double> b_matrix{18.0};
  std::vector<double> test_data{324.0};
  std::vector<double> out(kCount, 0);
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_matrix.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_matrix.data()));
  for (int i = 0; i < 4; ++i) {
    task_data_omp->inputs_count.emplace_back(kCount);
  }
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(kCount);

  sarafanov_m_canon_mat_mul_omp::CanonMatMulOMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  for (size_t i = 0; i < kCount; ++i) {
    EXPECT_NEAR(out[i], test_data[i], kInaccuracy);
  }
}

TEST(sarafanov_m_canon_mat_mul_omp, test_2x2_matrix) {
  constexpr size_t kCount = 2;
  constexpr double kInaccuracy = 0.001;
  std::vector<double> a_matrix{5.0, 6.0, 6.0, 5.0};
  std::vector<double> b_matrix{10.0, 2.0, 2.0, 10.0};
  std::vector<double> test_data{62.0, 70.0, 70.0, 62.0};
  std::vector<double> out(kCount * kCount, 0);
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_matrix.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_matrix.data()));
  for (int i = 0; i < 4; ++i) {
    task_data_omp->inputs_count.emplace_back(kCount);
  }
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  sarafanov_m_canon_mat_mul_omp::CanonMatMulOMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  for (size_t i = 0; i < kCount * kCount; ++i) {
    EXPECT_NEAR(out[i], test_data[i], kInaccuracy);
  }
}

TEST(sarafanov_m_canon_mat_mul_omp, test_3x3_matrix) {
  constexpr size_t kCount = 3;
  constexpr double kInaccuracy = 0.001;
  std::vector<double> a_matrix{1.0, 4.0, 8.0, 5.0, 6.0, 2.0, 2.0, 7.0, 7.0};
  std::vector<double> b_matrix{9.0, 1.0, 10.0, 12.0, 5.0, 2.0, 9.0, 7.0, 1.0};
  std::vector<double> test_data{129.0, 77.0, 26.0, 135.0, 49.0, 64.0, 165.0, 86.0, 41.0};
  std::vector<double> out(kCount * kCount, 0);
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_matrix.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_matrix.data()));
  for (int i = 0; i < 4; ++i) {
    task_data_omp->inputs_count.emplace_back(kCount);
  }
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  sarafanov_m_canon_mat_mul_omp::CanonMatMulOMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  for (size_t i = 0; i < kCount * kCount; ++i) {
    EXPECT_NEAR(out[i], test_data[i], kInaccuracy);
  }
}

TEST(sarafanov_m_canon_mat_mul_omp, test_random_5x5_matrix) {
  constexpr size_t kCount = 5;
  constexpr double kInaccuracy = 0.001;
  auto a_matrix = GenerateRandomData(static_cast<int>(kCount * kCount));
  auto single_matrix = GenerateSingleMatrix(static_cast<int>(kCount * kCount));
  std::vector<double> out(kCount * kCount, 0);
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_matrix.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(single_matrix.data()));
  for (int i = 0; i < 4; ++i) {
    task_data_omp->inputs_count.emplace_back(kCount);
  }
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  sarafanov_m_canon_mat_mul_omp::CanonMatMulOMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  for (size_t i = 0; i < kCount * kCount; ++i) {
    EXPECT_NEAR(out[i], a_matrix[i], kInaccuracy);
  }
}

TEST(sarafanov_m_canon_mat_mul_omp, test_random_30x30_matrix) {
  constexpr size_t kCount = 30;
  constexpr double kInaccuracy = 0.001;
  auto a_matrix = GenerateRandomData(static_cast<int>(kCount * kCount));
  auto single_matrix = GenerateSingleMatrix(static_cast<int>(kCount * kCount));
  std::vector<double> out(kCount * kCount, 0);
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_matrix.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(single_matrix.data()));
  for (int i = 0; i < 4; ++i) {
    task_data_omp->inputs_count.emplace_back(kCount);
  }
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  sarafanov_m_canon_mat_mul_omp::CanonMatMulOMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  for (size_t i = 0; i < kCount * kCount; ++i) {
    EXPECT_NEAR(out[i], a_matrix[i], kInaccuracy);
  }
}

TEST(sarafanov_m_canon_mat_mul_omp, test_3x2_matrix) {
  constexpr double kInaccuracy = 0.001;
  std::vector<double> a_matrix{1.0, 4.0, 8.0, 5.0, 6.0, 2.0};
  std::vector<double> b_matrix{9.0, 1.0, 10.0, 12.0, 5.0, 2.0};
  std::vector<double> test_data{57.0, 21.0, 18.0, 132.0, 33.0, 90.0, 78.0, 16.0, 64.0};
  std::vector<double> out(9, 0);
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_matrix.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_matrix.data()));
  task_data_omp->inputs_count.emplace_back(3);
  task_data_omp->inputs_count.emplace_back(2);
  task_data_omp->inputs_count.emplace_back(2);
  task_data_omp->inputs_count.emplace_back(3);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(9);

  sarafanov_m_canon_mat_mul_omp::CanonMatMulOMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], test_data[i], kInaccuracy);
  }
}

TEST(sarafanov_m_canon_mat_mul_omp, test_random_17x23_matrix) {
  constexpr int kRowsCount = 17;
  constexpr int kColumnsCount = 23;
  constexpr int kMaxSize = std::max(kRowsCount, kColumnsCount);
  constexpr double kInaccuracy = 0.001;
  auto a_matrix = GenerateRandomData(17 * 23);
  auto single_matrix = GenerateSingleMatrix(kMaxSize * kMaxSize);
  std::vector<double> out(kMaxSize * kMaxSize, 0);
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_matrix.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(single_matrix.data()));
  task_data_omp->inputs_count.emplace_back(kRowsCount);
  task_data_omp->inputs_count.emplace_back(kColumnsCount);
  task_data_omp->inputs_count.emplace_back(kMaxSize);
  task_data_omp->inputs_count.emplace_back(kMaxSize);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  sarafanov_m_canon_mat_mul_omp::CanonMatMulOMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.Validation(), true);
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();
  for (size_t i = 0; i < kRowsCount * kColumnsCount; ++i) {
    EXPECT_NEAR(out[i], a_matrix[i], kInaccuracy);
  }
}