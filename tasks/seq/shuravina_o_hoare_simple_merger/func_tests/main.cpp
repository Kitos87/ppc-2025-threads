#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/shuravina_o_hoare_simple_merger/include/ops_seq.hpp"

namespace {

bool IsPrime(size_t n) {
  if (n <= 1) {
    return false;
  }
  if (n <= 3) {
    return true;
  }
  if (n % 2 == 0 || n % 3 == 0) {
    return false;
  }
  for (size_t i = 5; i * i <= n; i += 6) {
    if (n % i == 0 || n % (i + 2) == 0) {
      return false;
    }
  }
  return true;
}

bool IsReverseSorted(const std::vector<int>& arr) {
  if (arr.empty()) {
    return true;
  }
  for (size_t i = 1; i < arr.size(); ++i) {
    if (arr[i - 1] < arr[i]) {
      return false;
    }
  }
  return true;
}

std::vector<int> GenerateRandomArray(size_t size, int min_val, int max_val) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(min_val, max_val);

  std::vector<int> arr(size);
  for (size_t i = 0; i < size; ++i) {
    arr[i] = distrib(gen);
  }
  return arr;
}

}  // namespace

TEST(shuravina_o_hoare_simple_merger, test_random_array) {
  const size_t array_size = 1000;
  const int min_val = -1000;
  const int max_val = 1000;

  std::vector<int> in = GenerateRandomArray(array_size, min_val, max_val);
  std::vector<int> out(in.size(), 0);

  ASSERT_FALSE(IsReverseSorted(in));

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  shuravina_o_hoare_simple_merger::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  for (size_t i = 1; i < out.size(); ++i) {
    ASSERT_LE(out[i - 1], out[i]);
  }
}

TEST(shuravina_o_hoare_simple_merger, test_large_random_array) {
  const size_t array_size = 10007;
  const int min_val = -10000;
  const int max_val = 10000;

  ASSERT_TRUE(IsPrime(array_size));

  std::vector<int> in = GenerateRandomArray(array_size, min_val, max_val);
  std::vector<int> out(in.size(), 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  shuravina_o_hoare_simple_merger::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  for (size_t i = 1; i < out.size(); ++i) {
    ASSERT_LE(out[i - 1], out[i]);
  }
}

TEST(shuravina_o_hoare_simple_merger, test_sort_and_merge) {
  std::vector<int> in = {5, 2, 9, 1, 5, 6};
  std::vector<int> out(in.size(), 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  shuravina_o_hoare_simple_merger::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  std::vector<int> expected = {1, 2, 5, 5, 6, 9};
  EXPECT_EQ(out, expected);
}

TEST(shuravina_o_hoare_simple_merger, test_empty_array) {
  std::vector<int> in = {};
  std::vector<int> out(in.size(), 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  shuravina_o_hoare_simple_merger::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  std::vector<int> expected = {};
  EXPECT_EQ(out, expected);
}

TEST(shuravina_o_hoare_simple_merger, test_single_element_array) {
  std::vector<int> in = {42};
  std::vector<int> out(in.size(), 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  shuravina_o_hoare_simple_merger::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  std::vector<int> expected = {42};
  EXPECT_EQ(out, expected);
}

TEST(shuravina_o_hoare_simple_merger, test_already_sorted_array) {
  std::vector<int> in = {1, 2, 3, 4, 5, 6};
  std::vector<int> out(in.size(), 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  shuravina_o_hoare_simple_merger::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  std::vector<int> expected = {1, 2, 3, 4, 5, 6};
  EXPECT_EQ(out, expected);
}

TEST(shuravina_o_hoare_simple_merger, test_array_with_negative_numbers) {
  std::vector<int> in = {-5, 2, -9, 1, 0, -3};
  std::vector<int> out(in.size(), 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  shuravina_o_hoare_simple_merger::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  std::vector<int> expected = {-9, -5, -3, 0, 1, 2};
  EXPECT_EQ(out, expected);
}