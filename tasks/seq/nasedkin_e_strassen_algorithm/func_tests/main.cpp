#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/nasedkin_e_strassen_algorithm/include/ops_seq.hpp"

namespace {
std::vector<double> GenerateRandomMatrix(int size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distrib(-100.0, 100.0);

  std::vector<double> matrix(size * size);
  for (int i = 0; i < size * size; ++i) {
    matrix[i] = distrib(gen);
  }
  return matrix;
}

void RunRandomMatrixTest(int size) {
  std::vector<double> in_a = GenerateRandomMatrix(size);
  std::vector<double> in_b = GenerateRandomMatrix(size);
  std::vector<double> out(size * size, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_b.data()));
  task_data_seq->inputs_count.emplace_back(in_a.size());
  task_data_seq->inputs_count.emplace_back(in_b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  nasedkin_e_strassen_algorithm_seq::StrassenSequential strassen_task_sequential(task_data_seq);
  ASSERT_EQ(strassen_task_sequential.Validation(), true);
  strassen_task_sequential.PreProcessing();
  strassen_task_sequential.Run();
  strassen_task_sequential.PostProcessing();
}

void RunFixedMatrixTest(int size) {
  std::vector<double> in_a(size * size);
  std::vector<double> in_b(size * size);

  for (int i = 0; i < size * size; ++i) {
    in_a[i] = static_cast<double>((size * size) - i);
  }

  for (int i = 0; i < size * size; ++i) {
    in_b[i] = static_cast<double>(i + 1);
  }

  std::vector<double> expected_result = nasedkin_e_strassen_algorithm_seq::StandardMultiply(in_a, in_b, size);
  std::vector<double> out(size * size, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_b.data()));
  task_data_seq->inputs_count.emplace_back(in_a.size());
  task_data_seq->inputs_count.emplace_back(in_b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  nasedkin_e_strassen_algorithm_seq::StrassenSequential strassen_task_sequential(task_data_seq);
  ASSERT_EQ(strassen_task_sequential.Validation(), true);
  strassen_task_sequential.PreProcessing();
  strassen_task_sequential.Run();
  strassen_task_sequential.PostProcessing();

  for (int i = 0; i < static_cast<int>(expected_result.size()); ++i) {
    EXPECT_NEAR(expected_result[i], out[i], 1e-6);
  }
}
}  // namespace

TEST(nasedkin_e_strassen_algorithm_seq, test_matrix_63x63_fixed) { RunFixedMatrixTest(63); }

TEST(nasedkin_e_strassen_algorithm_seq, test_matrix_64x64_fixed) { RunFixedMatrixTest(64); }

TEST(nasedkin_e_strassen_algorithm_seq, test_random_matrix_64x64) { RunRandomMatrixTest(64); }

TEST(nasedkin_e_strassen_algorithm_seq, test_random_matrix_127x127) { RunRandomMatrixTest(127); }

TEST(nasedkin_e_strassen_algorithm_seq, test_random_matrix_128x128) { RunRandomMatrixTest(128); }