#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/lavrentiev_A_CCS_SEQ/include/ops_seq.hpp"

namespace {
std::vector<double> GenerateRandomMatrix(int size, int sparse_size) {
  std::vector<double> data(size);
  std::random_device device;
  std::mt19937 generator(device());
  std::uniform_int_distribution<> random_element(-500, 500);
  size = size / sparse_size;
  for (int i = 0; i < size; ++i) {
    data[i] = static_cast<double>(random_element(generator));
  }
  std::ranges::shuffle(data, generator);
  return data;
}

std::vector<double> GenerateSingleMatrix(int size) {
  std::vector<double> test_data(size, 0.0);
  int sqrt = static_cast<int>(std::sqrt(size));
  for (int i = 0; i < sqrt; ++i) {
    for (int j = 0; j < sqrt; ++j) {
      if (i == j) {
        test_data[(sqrt * i) + j] = 1.0;
      }
    }
  }
  return test_data;
}
constexpr auto kEpsilon = 0.000001;
struct TestData {
  std::vector<double> random_data;
  std::vector<double> single_matrix;
  std::vector<double> result;
  std::shared_ptr<ppc::core::TaskData> task_data_seq;
  TestData(std::pair<int, int> matrix1_size, std::pair<int, int> matrix2_size, int sparse_size);
  [[nodiscard]] lavrentiev_a_ccs_seq::CCSSequential CreateTask() const;
};

TestData::TestData(std::pair<int, int> matrix1_size, std::pair<int, int> matrix2_size, int sparse_size) {
  random_data = GenerateRandomMatrix(matrix1_size.first * matrix1_size.second, sparse_size);
  single_matrix = GenerateSingleMatrix(matrix2_size.first * matrix2_size.second);
  result.resize(matrix1_size.first * matrix2_size.second);
  task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(random_data.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(single_matrix.data()));
  task_data_seq->inputs_count.emplace_back(matrix1_size.first);
  task_data_seq->inputs_count.emplace_back(matrix1_size.second);
  task_data_seq->inputs_count.emplace_back(matrix2_size.first);
  task_data_seq->inputs_count.emplace_back(matrix2_size.second);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data_seq->outputs_count.emplace_back(result.size());
}

lavrentiev_a_ccs_seq::CCSSequential TestData::CreateTask() const {
  return lavrentiev_a_ccs_seq::CCSSequential(task_data_seq);
}
}  // namespace

TEST(lavrentiev_a_ccs_seq, test_0x0_matrix) {
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> result;
  std::vector<double> test_result;
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  for (int i = 0; i < 4; ++i) {
    task_data_seq->inputs_count.emplace_back(0);
  }
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data_seq->outputs_count.emplace_back(result.size());

  lavrentiev_a_ccs_seq::CCSSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_NEAR(result[i], test_result[i], kEpsilon);
  }
}

TEST(lavrentiev_a_ccs_seq, test_3x2_matrix) {
  std::vector<double> a{2.0, 0.0, 0.0, 4.0, 0.0, 1.0};
  std::vector<double> b{6.0, 0.0, 0.0, 0.0, 0.0, 9.0};
  std::vector<double> result(3 * 3, 0.0);
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs_count.emplace_back(3);
  task_data_seq->inputs_count.emplace_back(2);
  task_data_seq->inputs_count.emplace_back(2);
  task_data_seq->inputs_count.emplace_back(3);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data_seq->outputs_count.emplace_back(result.size());

  lavrentiev_a_ccs_seq::CCSSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  std::vector<double> test_result{12.0, 0.0, 0.0, 0.0, 0.0, 36.0, 0.0, 0.0, 9.0};
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_NEAR(result[i], test_result[i], kEpsilon);
  }
}

TEST(lavrentiev_a_ccs_seq, test_3x3_matrixes) {
  std::vector<double> a{2.0, 0.0, 0.0, 0.0, 4.0, 1.0, 0.0, 1.0, 6.0};
  std::vector<double> b{6.0, 0.0, 0.0, 0.0, 0.0, 9.0, 7.0, 2.0, 0.0};
  std::vector<double> result(3 * 3, 0.0);
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  for (auto i = 0; i < 4; ++i) {
    task_data_seq->inputs_count.emplace_back(3);
  }
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data_seq->outputs_count.emplace_back(result.size());
  lavrentiev_a_ccs_seq::CCSSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  std::vector<double> test_result{12.0, 0.0, 0.0, 7.0, 2.0, 36.0, 42.0, 12.0, 9.0};
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_NEAR(result[i], test_result[i], kEpsilon);
  }
}

TEST(lavrentiev_a_ccs_seq, test_12x12_matrix) {
  auto task = TestData({12, 12}, {12, 12}, 1);
  auto test_task_sequential = task.CreateTask();
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (size_t i = 0; i < task.result.size(); ++i) {
    EXPECT_NEAR(task.result[i], task.random_data[i], kEpsilon);
  }
}

TEST(lavrentiev_a_ccs_seq, test_25x25_matrix) {
  auto task = TestData({25, 25}, {25, 25}, 2);
  auto test_task_sequential = task.CreateTask();
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (size_t i = 0; i < task.result.size(); ++i) {
    EXPECT_NEAR(task.result[i], task.random_data[i], kEpsilon);
  }
}