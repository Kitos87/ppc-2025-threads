#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/vavilov_v_cannon/include/ops_omp.hpp"

namespace {

using namespace vavilov_v_cannon_omp;

std::vector<double> GenerateRandomMatrix(int n, double min_val = -10.0, double max_val = 10.0) {
  std::vector<double> matrix(n * n);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(min_val, max_val);

  for (int i = 0; i < n * n; i++) {
    matrix[i] = dist(gen);
  }
  return matrix;
}

std::vector<double> MultMat(const std::vector<double>& a, const std::vector<double>& b, int n) {
  std::vector<double> c(n * n, 0.0);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        c[(i * n) + j] += a[(i * n) + k] * b[(k * n) + j];
      }
    }
  }
  return c;
}

TEST(vavilov_v_cannon_omp, test_random) {
  constexpr int kN = 16;
  constexpr int kNumblocks = 4;
  auto a = GenerateRandomMatrix(kN);
  auto b = GenerateRandomMatrix(kN);
  std::vector<double> expected_output = MultMat(a, b, kN);
  std::vector<double> c(kN * kN, 0.0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_omp->inputs_count.emplace_back(a.size());
  task_data_omp->inputs_count.emplace_back(b.size());
  task_data_omp->inputs_count.emplace_back(kNumblocks);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(c.data()));
  task_data_omp->outputs_count.emplace_back(c.size());

  vavilov_v_cannon_omp::CannonOMP task_omp(task_data_omp);
  ASSERT_TRUE(task_omp.Validation());
  task_omp.PreProcessing();
  task_omp.Run();
  task_omp.PostProcessing();

  for (int i = 0; i < kN * kN; i++) {
    EXPECT_NEAR(expected_output[i], c[i], 1e-6);
  }
}

TEST(vavilov_v_cannon_omp, test_fixed_4x4) {
  constexpr int kN = 4;
  constexpr int kNumblocks = 2;
  std::vector<double> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<double> b = {1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0};
  std::vector<double> expected_output = {4, 6, 6, 4, 12, 14, 14, 12, 20, 22, 22, 20, 28, 30, 30, 28};
  std::vector<double> c(kN * kN, 0.0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_omp->inputs_count.emplace_back(a.size());
  task_data_omp->inputs_count.emplace_back(b.size());
  task_data_omp->inputs_count.emplace_back(kNumblocks);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(c.data()));
  task_data_omp->outputs_count.emplace_back(c.size());

  vavilov_v_cannon_omp::CannonOMP task_omp(task_data_omp);
  ASSERT_TRUE(task_omp.Validation());
  task_omp.PreProcessing();
  task_omp.Run();
  task_omp.PostProcessing();

  for (int i = 0; i < kN * kN; i++) {
    EXPECT_EQ(expected_output[i], c[i]);
  }
}

TEST(vavilov_v_cannon_omp, test_invalid_size_1) {
  std::vector<double> a(2 * 2, 1.0);
  std::vector<double> b(3 * 2, 1.0);
  std::vector<double> c(2 * 2, 0.0);
  constexpr int kNumblocks = 1;

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_omp->inputs_count.emplace_back(a.size());
  task_data_omp->inputs_count.emplace_back(b.size());
  task_data_omp->inputs_count.emplace_back(kNumblocks);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(c.data()));
  task_data_omp->outputs_count.emplace_back(c.size());

  vavilov_v_cannon_omp::CannonOMP task_omp(task_data_omp);
  ASSERT_FALSE(task_omp.Validation());
}

TEST(vavilov_v_cannon_omp, test_225) {
  constexpr int kN = 36;
  constexpr int kNumblocks = 6;
  std::vector<double> a(kN * kN, 1.0);
  std::vector<double> b(kN * kN, 1.0);
  std::vector<double> c(kN * kN, 0.0);
  std::vector<double> expected_output(kN * kN, kN);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_omp->inputs_count.emplace_back(a.size());
  task_data_omp->inputs_count.emplace_back(b.size());
  task_data_omp->inputs_count.emplace_back(kNumblocks);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(c.data()));
  task_data_omp->outputs_count.emplace_back(c.size());

  vavilov_v_cannon_omp::CannonOMP task_omp(task_data_omp);
  ASSERT_TRUE(task_omp.Validation());
  task_omp.PreProcessing();
  task_omp.Run();
  task_omp.PostProcessing();

  for (int i = 0; i < kN * kN; i++) {
    EXPECT_EQ(expected_output[i], c[i]);
  }
}

TEST(vavilov_v_cannon_omp, test_identity_matrix) {
  constexpr int kN = 36;
  constexpr int kNumblocks = 6;
  std::vector<double> a(kN * kN, 1.0);
  std::vector<double> b(kN * kN, 0.0);
  std::vector<double> c(kN * kN, 0.0);

  for (int i = 0; i < kN; i++) {
    b[(i * kN) + i] = 1.0;
  }

  auto expected_output = a;

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_omp->inputs_count.emplace_back(a.size());
  task_data_omp->inputs_count.emplace_back(b.size());
  task_data_omp->inputs_count.emplace_back(kNumblocks);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(c.data()));
  task_data_omp->outputs_count.emplace_back(c.size());

  vavilov_v_cannon_omp::CannonOMP task_omp(task_data_omp);
  ASSERT_TRUE(task_omp.Validation());
  task_omp.PreProcessing();
  task_omp.Run();
  task_omp.PostProcessing();

  for (int i = 0; i < kN * kN; i++) {
    EXPECT_EQ(expected_output[i], c[i]);
  }
}

TEST(vavilov_v_cannon_omp, test_zero_matrix) {
  constexpr int kN = 36;
  constexpr int kNumblocks = 6;
  std::vector<double> a(kN * kN, 1.0);
  std::vector<double> b(kN * kN, 0.0);
  std::vector<double> c(kN * kN, 0.0);
  std::vector<double> expected_output(kN * kN, 0.0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_omp->inputs_count.emplace_back(a.size());
  task_data_omp->inputs_count.emplace_back(b.size());
  task_data_omp->inputs_count.emplace_back(kNumblocks);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(c.data()));
  task_data_omp->outputs_count.emplace_back(c.size());

  vavilov_v_cannon_omp::CannonOMP task_omp(task_data_omp);
  ASSERT_TRUE(task_omp.Validation());
  task_omp.PreProcessing();
  task_omp.Run();
  task_omp.PostProcessing();

  for (int i = 0; i < kN * kN; i++) {
    EXPECT_EQ(expected_output[i], c[i]);
  }
}

TEST(vavilov_v_cannon_omp, test_15) {
  constexpr int kN = 15;
  constexpr int kNumblocks = 3;
  std::vector<double> a(kN * kN, 1.0);
  std::vector<double> b(kN * kN, 1.0);
  std::vector<double> c(kN * kN, 0.0);
  std::vector<double> expected_output(kN * kN, kN);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_omp->inputs_count.emplace_back(a.size());
  task_data_omp->inputs_count.emplace_back(b.size());
  task_data_omp->inputs_count.emplace_back(kNumblocks);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(c.data()));
  task_data_omp->outputs_count.emplace_back(c.size());

  vavilov_v_cannon_omp::CannonOMP task_omp(task_data_omp);
  ASSERT_TRUE(task_omp.Validation());
  task_omp.PreProcessing();
  task_omp.Run();
  task_omp.PostProcessing();

  for (int i = 0; i < kN * kN; i++) {
    EXPECT_EQ(expected_output[i], c[i]);
  }
}

TEST(vavilov_v_cannon_onp, test_15_error_validation) {
  constexpr int kN = 15;
  constexpr int kNumblocks = 4;
  std::vector<double> a(kN * kN, 1.0);
  std::vector<double> b(kN * kN, 1.0);
  std::vector<double> c(kN * kN, 0.0);
  std::vector<double> expected_output(kN * kN, kN);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_omp->inputs_count.emplace_back(a.size());
  task_data_omp->inputs_count.emplace_back(b.size());
  task_data_omp->inputs_count.emplace_back(kNumblocks);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(c.data()));
  task_data_omp->outputs_count.emplace_back(c.size());

  vavilov_v_cannon_omp::CannonOMP task_omp(task_data_omp);
  ASSERT_FALSE(task_omp.Validation());
}
}  // namespace
