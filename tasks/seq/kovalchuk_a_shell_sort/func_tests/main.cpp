#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/kovalchuk_a_shell_sort/include/ops_seq.hpp"

TEST(kovalchuk_a_shell_sort, test_sort_basic) {
  std::vector<int> input = {9, 2, 5, 1, 7};
  std::vector<int> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<kovalchuk_a_shell_sort::ShellSortSequential>(task_data);

  ASSERT_TRUE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();

  std::vector<int> expected = {1, 2, 5, 7, 9};
  ASSERT_EQ(expected, output);
}

TEST(kovalchuk_a_shell_sort, test_empty_array) {
  std::vector<int> input;
  std::vector<int> output;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<kovalchuk_a_shell_sort::ShellSortSequential>(task_data);

  ASSERT_TRUE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();

  ASSERT_TRUE(output.empty());
}

TEST(kovalchuk_a_shell_sort, test_single_element) {
  std::vector<int> input = {42};
  std::vector<int> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<kovalchuk_a_shell_sort::ShellSortSequential>(task_data);

  ASSERT_TRUE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();

  ASSERT_EQ(input, output);
}

TEST(kovalchuk_a_shell_sort, test_already_sorted) {
  std::vector<int> input = {1, 2, 3, 4, 5};
  std::vector<int> output(input.size());
  std::vector<int> expected = input;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<kovalchuk_a_shell_sort::ShellSortSequential>(task_data);

  ASSERT_TRUE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();

  ASSERT_EQ(expected, output);
}

TEST(kovalchuk_a_shell_sort, test_reverse_sorted) {
  std::vector<int> input = {9, 7, 5, 3, 1};
  std::vector<int> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<kovalchuk_a_shell_sort::ShellSortSequential>(task_data);

  ASSERT_TRUE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();

  std::vector<int> expected = {1, 3, 5, 7, 9};
  ASSERT_EQ(expected, output);
}

TEST(kovalchuk_a_shell_sort, test_duplicates) {
  std::vector<int> input = {5, 2, 5, 1, 2};
  std::vector<int> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<kovalchuk_a_shell_sort::ShellSortSequential>(task_data);

  ASSERT_TRUE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();

  std::vector<int> expected = {1, 2, 2, 5, 5};
  ASSERT_EQ(expected, output);
}

TEST(kovalchuk_a_shell_sort, test_negative_numbers) {
  std::vector<int> input = {-5, 0, -3, 10, -1};
  std::vector<int> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<kovalchuk_a_shell_sort::ShellSortSequential>(task_data);

  ASSERT_TRUE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();

  std::vector<int> expected = {-5, -3, -1, 0, 10};
  ASSERT_EQ(expected, output);
}

TEST(kovalchuk_a_shell_sort, test_validation_fail) {
  std::vector<int> input = {1, 2, 3};
  std::vector<int> output(5);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<kovalchuk_a_shell_sort::ShellSortSequential>(task_data);

  ASSERT_FALSE(task->Validation());
}

TEST(kovalchuk_a_shell_sort, test_large_values) {
  std::vector<int> input = {INT32_MAX, 0, INT32_MIN};
  std::vector<int> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<kovalchuk_a_shell_sort::ShellSortSequential>(task_data);

  ASSERT_TRUE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();

  std::vector<int> expected = {INT32_MIN, 0, INT32_MAX};
  ASSERT_EQ(expected, output);
}

TEST(kovalchuk_a_shell_sort, test_double_reverse) {
  std::vector<int> input = {5, 4, 3, 2, 1, 5, 4, 3, 2, 1};
  std::vector<int> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<kovalchuk_a_shell_sort::ShellSortSequential>(task_data);

  ASSERT_TRUE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();

  std::vector<int> expected = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
  ASSERT_EQ(expected, output);
}

TEST(kovalchuk_a_shell_sort, test_double_reverse_size6) {
  std::vector<int> input = {3, 2, 1, 3, 2, 1};
  std::vector<int> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<kovalchuk_a_shell_sort::ShellSortSequential>(task_data);

  ASSERT_TRUE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();

  std::vector<int> expected = {1, 1, 2, 2, 3, 3};
  ASSERT_EQ(expected, output);
}

TEST(kovalchuk_a_shell_sort, test_double_reverse_size8) {
  std::vector<int> input = {4, 3, 2, 1, 4, 3, 2, 1};
  std::vector<int> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<kovalchuk_a_shell_sort::ShellSortSequential>(task_data);

  ASSERT_TRUE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();

  std::vector<int> expected = {1, 1, 2, 2, 3, 3, 4, 4};
  ASSERT_EQ(expected, output);
}