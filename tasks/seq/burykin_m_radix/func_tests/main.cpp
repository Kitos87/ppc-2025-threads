#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/burykin_m_radix/include/ops_seq.hpp"

namespace {

std::vector<int> GenerateRandomVector(size_t size, int min_val = -1000, int max_val = 1000) {
  std::vector<int> vec(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(min_val, max_val);
  for (auto& elem : vec) {
    elem = dis(gen);
  }
  return vec;
}

}  // namespace

TEST(burykin_m_radix_seq, AlreadySorted) {
  std::vector<int> input = {-5, -3, 0, 2, 3, 10};
  std::vector<int> expected = input;
  std::vector<int> output(input.size(), 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.push_back(static_cast<std::uint32_t>(input.size()));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.push_back(static_cast<std::uint32_t>(output.size()));

  burykin_m_radix_seq::RadixSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  EXPECT_EQ(output, expected);
}

TEST(burykin_m_radix_seq, ReverseSorted) {
  std::vector<int> input = {10, 3, 2, 0, -3, -5};
  std::vector<int> expected = input;
  std::ranges::sort(expected);
  std::vector<int> output(input.size(), 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.push_back(static_cast<std::uint32_t>(input.size()));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.push_back(static_cast<std::uint32_t>(output.size()));

  burykin_m_radix_seq::RadixSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  EXPECT_EQ(output, expected);
}

TEST(burykin_m_radix_seq, RandomVector) {
  constexpr size_t kSize = 1000;
  std::vector<int> input = GenerateRandomVector(kSize);
  std::vector<int> expected = input;
  std::ranges::sort(expected);
  std::vector<int> output(kSize, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.push_back(static_cast<std::uint32_t>(input.size()));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.push_back(static_cast<std::uint32_t>(output.size()));

  burykin_m_radix_seq::RadixSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  EXPECT_EQ(output, expected);
}

TEST(burykin_m_radix_seq, AllEqual) {
  constexpr size_t kSize = 100;
  std::vector<int> input(kSize, 42);
  std::vector<int> expected = input;
  std::vector<int> output(kSize, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.push_back(static_cast<std::uint32_t>(input.size()));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.push_back(static_cast<std::uint32_t>(output.size()));

  burykin_m_radix_seq::RadixSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  EXPECT_EQ(output, expected);
}

TEST(burykin_m_radix_seq, EmptyVector) {
  std::vector<int> input;
  std::vector<int> expected;
  std::vector<int> output;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.push_back(static_cast<std::uint32_t>(input.size()));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.push_back(static_cast<std::uint32_t>(output.size()));

  burykin_m_radix_seq::RadixSequential task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  EXPECT_EQ(output, expected);
}
