#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/gromov_a_fox_algorithm/include/ops_seq.hpp"

namespace {
std::vector<double> GenerateRandomMatrix(size_t n, double min_val, double max_val) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(min_val, max_val);

  std::vector<double> matrix(n * n);
  for (size_t i = 0; i < n * n; ++i) {
    matrix[i] = dis(gen);
  }
  return matrix;
}
}  // namespace

TEST(gromov_a_fox_algorithm_seq, test_random_5x5) {
  constexpr size_t kN = 5;

  std::vector<double> a = GenerateRandomMatrix(kN, -10.0, 10.0);
  std::vector<double> b = GenerateRandomMatrix(kN, -10.0, 10.0);
  std::vector<double> out(kN * kN, 0.0);

  std::vector<double> expected(kN * kN, 0.0);
  for (size_t i = 0; i < kN; ++i) {
    for (size_t j = 0; j < kN; ++j) {
      for (size_t k = 0; k < kN; ++k) {
        expected[(i * kN) + j] += a[(i * kN) + k] * b[(k * kN) + j];
      }
    }
  }

  std::vector<double> input;
  input.reserve(a.size() + b.size());
  std::ranges::copy(a, std::back_inserter(input));
  std::ranges::copy(b, std::back_inserter(input));

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  gromov_a_fox_algorithm_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], expected[i], 1e-9);
  }
}

TEST(gromov_a_fox_algorithm_seq, test_random_10x10) {
  constexpr size_t kN = 10;

  std::vector<double> a = GenerateRandomMatrix(kN, -10.0, 10.0);
  std::vector<double> b = GenerateRandomMatrix(kN, -10.0, 10.0);
  std::vector<double> out(kN * kN, 0.0);

  std::vector<double> expected(kN * kN, 0.0);
  for (size_t i = 0; i < kN; ++i) {
    for (size_t j = 0; j < kN; ++j) {
      for (size_t k = 0; k < kN; ++k) {
        expected[(i * kN) + j] += a[(i * kN) + k] * b[(k * kN) + j];
      }
    }
  }

  std::vector<double> input;
  input.reserve(a.size() + b.size());
  std::ranges::copy(a, std::back_inserter(input));
  std::ranges::copy(b, std::back_inserter(input));

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  gromov_a_fox_algorithm_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], expected[i], 1e-9);
  }
}

TEST(gromov_a_fox_algorithm_seq, test_4x4) {
  constexpr size_t kN = 4;

  std::vector<double> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
  std::vector<double> b = {16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
  std::vector<double> out(kN * kN, 0.0);

  std::vector<double> expected = {80.0,  70.0,  60.0,  50.0,  240.0, 214.0, 188.0, 162.0,
                                  400.0, 358.0, 316.0, 274.0, 560.0, 502.0, 444.0, 386.0};

  std::vector<double> input;
  input.reserve(a.size() + b.size());

  std::ranges::copy(a, std::back_inserter(input));
  std::ranges::copy(b, std::back_inserter(input));

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  gromov_a_fox_algorithm_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], expected[i], 1e-9);
  }
}

TEST(gromov_a_fox_algorithm_seq, test_50x50) {
  constexpr size_t kN = 50;

  std::vector<double> a(kN * kN, 0.0);
  std::vector<double> b(kN * kN, 0.0);
  std::vector<double> out(kN * kN, 0.0);

  for (size_t i = 0; i < kN; ++i) {
    for (size_t j = 0; j < kN; ++j) {
      a[(i * kN) + j] = static_cast<double>((i * kN) + j + 1);
      b[(i * kN) + j] = static_cast<double>((kN * kN) - (i * kN + j));
    }
  }

  std::vector<double> input;
  input.reserve(a.size() + b.size());

  std::ranges::copy(a, std::back_inserter(input));
  std::ranges::copy(b, std::back_inserter(input));

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  gromov_a_fox_algorithm_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  std::vector<double> expected(kN * kN, 0.0);
  for (size_t i = 0; i < kN; ++i) {
    for (size_t j = 0; j < kN; ++j) {
      for (size_t k = 0; k < kN; ++k) {
        expected[(i * kN) + j] += a[(i * kN) + k] * b[(k * kN) + j];
      }
    }
  }

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], expected[i], 1e-9);
  }
}

TEST(gromov_a_fox_algorithm_seq, identity_3x3) {
  constexpr size_t kN = 3;

  std::vector<double> a = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  std::vector<double> b = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  std::vector<double> out(kN * kN, 0.0);

  std::vector<double> expected = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

  std::vector<double> input;
  input.reserve(a.size() + b.size());

  std::ranges::copy(a, std::back_inserter(input));
  std::ranges::copy(b, std::back_inserter(input));

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  gromov_a_fox_algorithm_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], expected[i], 1e-9);
  }
}

TEST(gromov_a_fox_algorithm_seq, test_run_small_matrix) {
  constexpr size_t kN = 4;

  std::vector<double> a(kN * kN, 1.0);
  std::vector<double> b(kN * kN, 1.0);
  std::vector<double> out(kN * kN, 0.0);

  std::vector<double> input;
  input.reserve(a.size() + b.size());

  std::ranges::copy(a, std::back_inserter(input));
  std::ranges::copy(b, std::back_inserter(input));

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  gromov_a_fox_algorithm_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  std::vector<double> expected(kN * kN, static_cast<double>(kN));
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], expected[i], 1e-9);
  }
}

TEST(gromov_a_fox_algorithm_seq, test_zero_matrix_4x4) {
  constexpr size_t kN = 4;

  std::vector<double> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
  std::vector<double> b(kN * kN, 0.0);
  std::vector<double> out(kN * kN, 0.0);

  std::vector<double> expected(kN * kN, 0.0);

  std::vector<double> input;
  input.reserve(a.size() + b.size());

  std::ranges::copy(a, std::back_inserter(input));
  std::ranges::copy(b, std::back_inserter(input));

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  gromov_a_fox_algorithm_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], expected[i], 1e-9);
  }
}

TEST(gromov_a_fox_algorithm_seq, test_negative_values_3x3) {
  constexpr size_t kN = 3;

  std::vector<double> a = {-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0};
  std::vector<double> b = {1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0};
  std::vector<double> out(kN * kN, 0.0);

  std::vector<double> expected = {-30.0, 36.0, -42.0, 66.0, -81.0, 96.0, -102.0, 126.0, -150.0};

  std::vector<double> input;
  input.reserve(a.size() + b.size());

  std::ranges::copy(a, std::back_inserter(input));
  std::ranges::copy(b, std::back_inserter(input));

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  gromov_a_fox_algorithm_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], expected[i], 1e-9);
  }
}

TEST(gromov_a_fox_algorithm_seq, identity_times_arbitrary_3x3) {
  constexpr size_t kN = 3;
  std::vector<double> a = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};  // I
  std::vector<double> b = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};  // A
  std::vector<double> out(kN * kN, 0.0);
  std::vector<double> expected = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};  // A

  std::vector<double> input;
  input.reserve(a.size() + b.size());
  std::ranges::copy(a, std::back_inserter(input));
  std::ranges::copy(b, std::back_inserter(input));

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  gromov_a_fox_algorithm_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], expected[i], 1e-9);
  }
}