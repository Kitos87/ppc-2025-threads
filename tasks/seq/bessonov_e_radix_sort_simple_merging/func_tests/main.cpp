#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/bessonov_e_radix_sort_simple_merging/include/ops_seq.hpp"

namespace {
std::vector<double> GenerateVector(std::size_t n, double first, double last) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(first, last);

  std::vector<double> result(n);
  for (std::size_t i = 0; i < n; ++i) {
    result[i] = dist(gen);
  }
  return result;
}
}  // namespace

TEST(bessonov_e_radix_sort_simple_merging_seq, FirstTest) {
  std::vector<double> input_vector = {3.4, 1.2, 0.5, 7.8, 2.3, 4.5, 6.7, 8.9, 1.0, 0.2, 5.6, 4.3, 9.1, 1.5, 3.0};
  std::vector<double> output_vector(input_vector.size(), 0.0);
  std::vector<double> result_vector = {0.2, 0.5, 1.0, 1.2, 1.5, 2.3, 3.0, 3.4, 4.3, 4.5, 5.6, 6.7, 7.8, 8.9, 9.1};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
  task_data->outputs_count.emplace_back(output_vector.size());

  bessonov_e_radix_sort_simple_merging_seq::TestTaskSequential test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  ASSERT_EQ(output_vector, result_vector);
}

TEST(bessonov_e_radix_sort_simple_merging_seq, SingleElementTest) {
  std::vector<double> input_vector = {42.0};
  std::vector<double> output_vector(1, 0.0);
  std::vector<double> result_vector = {42.0};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
  task_data->outputs_count.emplace_back(output_vector.size());

  bessonov_e_radix_sort_simple_merging_seq::TestTaskSequential test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  ASSERT_EQ(output_vector, result_vector);
}

TEST(bessonov_e_radix_sort_simple_merging_seq, NegativeAndPositiveTest) {
  std::vector<double> input_vector = {-3.2, 1.1, -7.5, 0.0, 4.4, -2.2, 3.3};
  std::vector<double> output_vector(input_vector.size(), 0.0);
  std::vector<double> result_vector = {-7.5, -3.2, -2.2, 0.0, 1.1, 3.3, 4.4};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
  task_data->outputs_count.emplace_back(output_vector.size());

  bessonov_e_radix_sort_simple_merging_seq::TestTaskSequential test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  ASSERT_EQ(output_vector, result_vector);
}

TEST(bessonov_e_radix_sort_simple_merging_seq, RandomVectorTest) {
  const std::size_t n = 1000;
  std::vector<double> input_vector = GenerateVector(n, -1000.0, 1000.0);
  std::vector<double> output_vector(n, 0.0);

  std::vector<double> result_vector = input_vector;
  std::ranges::sort(result_vector);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
  task_data->outputs_count.emplace_back(output_vector.size());

  bessonov_e_radix_sort_simple_merging_seq::TestTaskSequential test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  ASSERT_EQ(output_vector, result_vector);
}

TEST(bessonov_e_radix_sort_simple_merging_seq, AllSameElementsTest) {
  std::vector<double> input_vector = {3.14, 3.14, 3.14, 3.14};
  std::vector<double> output_vector(input_vector.size(), 0.0);
  std::vector<double> result_vector = {3.14, 3.14, 3.14, 3.14};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
  task_data->outputs_count.emplace_back(output_vector.size());

  bessonov_e_radix_sort_simple_merging_seq::TestTaskSequential test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  ASSERT_EQ(output_vector, result_vector);
}

TEST(bessonov_e_radix_sort_simple_merging_seq, ExtremeValuesTest) {
  std::vector<double> input_vector = {std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(), 0.0,
                                      -42.5, 100.0};
  std::vector<double> output_vector(input_vector.size(), 0.0);
  std::vector<double> result_vector = {std::numeric_limits<double>::lowest(), -42.5, 0.0, 100.0,
                                       std::numeric_limits<double>::max()};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
  task_data->outputs_count.emplace_back(output_vector.size());

  bessonov_e_radix_sort_simple_merging_seq::TestTaskSequential test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  ASSERT_EQ(output_vector, result_vector);
}

TEST(bessonov_e_radix_sort_simple_merging_seq, TinyNumbersTest) {
  std::vector<double> input_vector = {1e-10, -1e-10, 1e-20, -1e-20};
  std::vector<double> output_vector(input_vector.size(), 0.0);
  std::vector<double> result_vector = {-1e-10, -1e-20, 1e-20, 1e-10};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
  task_data->outputs_count.emplace_back(output_vector.size());

  bessonov_e_radix_sort_simple_merging_seq::TestTaskSequential test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  ASSERT_EQ(output_vector, result_vector);
}

TEST(bessonov_e_radix_sort_simple_merging_seq, InvalidInputOutputSizeTest) {
  std::vector<double> input = {1.0, 2.0, 3.0};
  std::vector<double> output(2, 0.0);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  bessonov_e_radix_sort_simple_merging_seq::TestTaskSequential test_task(task_data);
  ASSERT_FALSE(test_task.Validation());
}

TEST(bessonov_e_radix_sort_simple_merging_seq, ValidationEmptyTest) {
  std::vector<double> input_vector;
  std::vector<double> output_vector;
  std::vector<double> result_vector;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
  task_data->outputs_count.emplace_back(output_vector.size());

  bessonov_e_radix_sort_simple_merging_seq::TestTaskSequential test_task(task_data);
  ASSERT_FALSE(test_task.Validation());
}

TEST(bessonov_e_radix_sort_simple_merging_seq, ReverseOrderTest) {
  std::vector<double> input_vector = {9.1, 8.9, 7.8, 6.7, 5.6, 4.5, 4.3, 3.4, 3.0, 2.3, 1.5, 1.2, 1.0, 0.5, 0.2};
  std::vector<double> output_vector(input_vector.size(), 0.0);
  std::vector<double> result_vector = {0.2, 0.5, 1.0, 1.2, 1.5, 2.3, 3.0, 3.4, 4.3, 4.5, 5.6, 6.7, 7.8, 8.9, 9.1};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
  task_data->outputs_count.emplace_back(output_vector.size());

  bessonov_e_radix_sort_simple_merging_seq::TestTaskSequential test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  ASSERT_EQ(output_vector, result_vector);
}