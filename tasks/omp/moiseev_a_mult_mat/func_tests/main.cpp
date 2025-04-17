#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/moiseev_a_mult_mat/include/ops_omp.hpp"

namespace {

std::vector<double> GenerateRandomMatrix(size_t rows, size_t cols) {
  std::vector<double> matrix(rows * cols);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-100.0, 100.0);

  for (auto &val : matrix) {
    val = dist(gen);
  }
  return matrix;
}

}  // namespace

TEST(moiseev_a_mult_mat_omp, test_large_matrix) {
  constexpr size_t kSize = 100;
  auto a = GenerateRandomMatrix(kSize, kSize);
  auto b = GenerateRandomMatrix(kSize, kSize);
  std::vector<double> c(kSize * kSize, 0.0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_omp->inputs_count.emplace_back(a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_omp->inputs_count.emplace_back(b.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data_omp->outputs_count.emplace_back(c.size());

  moiseev_a_mult_mat_omp::MultMatOMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.Validation());
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  EXPECT_FALSE(c.empty());
}

TEST(moiseev_a_mult_mat_omp, test_small_matrix) {
  auto a = GenerateRandomMatrix(2, 2);
  auto b = GenerateRandomMatrix(2, 2);
  std::vector<double> c(2 * 2, 0.0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_omp->inputs_count.emplace_back(a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_omp->inputs_count.emplace_back(b.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data_omp->outputs_count.emplace_back(c.size());

  moiseev_a_mult_mat_omp::MultMatOMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.Validation());
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  EXPECT_FALSE(c.empty());
}

TEST(moiseev_a_mult_mat_omp, test_matrix) {
  constexpr size_t kSize = 5;
  auto a = GenerateRandomMatrix(kSize, kSize);
  auto b = GenerateRandomMatrix(kSize, kSize);
  std::vector<double> c(kSize * kSize, 0.0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_omp->inputs_count.emplace_back(a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_omp->inputs_count.emplace_back(b.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data_omp->outputs_count.emplace_back(c.size());

  moiseev_a_mult_mat_omp::MultMatOMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.Validation());
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  EXPECT_FALSE(c.empty());
}

TEST(moiseev_a_mult_mat_omp, test_zero_matrix) {
  constexpr size_t kSize = 4;
  auto a = GenerateRandomMatrix(kSize, kSize);
  std::vector<double> b(kSize * kSize, 0.0);
  std::vector<double> c(kSize * kSize, 0.0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_omp->inputs_count.emplace_back(a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_omp->inputs_count.emplace_back(b.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data_omp->outputs_count.emplace_back(c.size());

  moiseev_a_mult_mat_omp::MultMatOMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.Validation());
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  EXPECT_EQ(c, b);
}

TEST(moiseev_a_mult_mat_omp, test_known_result) {
  constexpr size_t kSize = 2;
  std::vector<double> a = {1, 2, 3, 4};
  std::vector<double> b = {5, 6, 7, 8};
  std::vector<double> expected_c = {19, 22, 43, 50};

  std::vector<double> c(kSize * kSize, 0.0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_omp->inputs_count.emplace_back(a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_omp->inputs_count.emplace_back(b.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data_omp->outputs_count.emplace_back(c.size());

  moiseev_a_mult_mat_omp::MultMatOMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.Validation());
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  EXPECT_EQ(c, expected_c);
}

TEST(moiseev_a_mult_mat_omp, test_repeated_values) {
  constexpr size_t kSize = 2;
  std::vector<double> a = {2, 2, 2, 2};
  std::vector<double> b = {3, 3, 3, 3};
  std::vector<double> expected_c = {12, 12, 12, 12};

  std::vector<double> c(kSize * kSize, 0.0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_omp->inputs_count.emplace_back(a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_omp->inputs_count.emplace_back(b.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data_omp->outputs_count.emplace_back(c.size());

  moiseev_a_mult_mat_omp::MultMatOMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.Validation());
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  EXPECT_EQ(c, expected_c);
}

TEST(moiseev_a_mult_mat_omp, test_prime_values) {
  constexpr size_t kSize = 2;
  std::vector<double> a = {5, 7, 11, 13};
  std::vector<double> b = {17, 19, 23, 29};
  std::vector<double> expected_c = {246, 298, 486, 586};

  std::vector<double> c(kSize * kSize, 0.0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_omp->inputs_count.emplace_back(a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_omp->inputs_count.emplace_back(b.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data_omp->outputs_count.emplace_back(c.size());

  moiseev_a_mult_mat_omp::MultMatOMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.Validation());
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  EXPECT_EQ(c, expected_c);
}

TEST(moiseev_a_mult_mat_omp, test_negative_values) {
  constexpr size_t kSize = 3;
  std::vector<double> a = {-1, 2, -3, 4, -5, 6, -7, 8, -9};
  std::vector<double> b = {9, -8, 7, -6, 5, -4, 3, -2, 1};
  std::vector<double> expected_c = {-30, 24, -18, 84, -69, 54, -138, 114, -90};

  std::vector<double> c(kSize * kSize, 0.0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_omp->inputs_count.emplace_back(a.size());
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_omp->inputs_count.emplace_back(b.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data_omp->outputs_count.emplace_back(c.size());

  moiseev_a_mult_mat_omp::MultMatOMP test_task_omp(task_data_omp);
  ASSERT_TRUE(test_task_omp.Validation());
  test_task_omp.PreProcessing();
  test_task_omp.Run();
  test_task_omp.PostProcessing();

  EXPECT_EQ(c, expected_c);
}
