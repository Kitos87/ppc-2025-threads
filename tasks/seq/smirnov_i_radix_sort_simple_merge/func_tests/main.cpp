#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/smirnov_i_radix_sort_simple_merge/include/ops_seq.hpp"

TEST(smirnov_i_radix_sort_simple_merge_seq, test_wrong_size) {
  // Create data
  std::vector<int> in(2, 0);
  std::vector<int> exp_out(2, 0);
  std::vector<int> out(1);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  smirnov_i_radix_sort_simple_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}
TEST(smirnov_i_radix_sort_simple_merge_seq, test_scalar) {
  constexpr size_t kCount = 1;

  // Create data
  std::vector<int> in(kCount, 0);
  std::vector<int> exp_out(kCount, 0);
  std::vector<int> out(kCount);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  smirnov_i_radix_sort_simple_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(exp_out, out);
}

TEST(smirnov_i_radix_sort_simple_merge_seq, test_17_elem) {
  // Create data
  std::vector<int> in{6, 134, 0, 6, 7, 1, 2, 4, 5, 3268, 6, 1, 8, 4, 234, 123120, 4};
  std::vector<int> exp_out{0, 1, 1, 2, 4, 4, 4, 5, 6, 6, 6, 7, 8, 134, 234, 3268, 123120};
  std::vector<int> out(17);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  smirnov_i_radix_sort_simple_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(exp_out, out);
}

TEST(smirnov_i_radix_sort_simple_merge_seq, test_10_elem) {
  std::vector<int> in{1, 6, 6, 1, 8, 1, 8, 1, 8, 1};
  std::vector<int> exp_out{1, 1, 1, 1, 1, 6, 6, 8, 8, 8};
  std::vector<int> out(10);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  smirnov_i_radix_sort_simple_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(exp_out, out);
}

TEST(smirnov_i_radix_sort_simple_merge_seq, test_256_elem_sorted) {
  constexpr size_t kCount = 256;

  // Create data
  std::vector<int> in(kCount, 0);
  std::vector<int> exp_out(kCount, 0);
  std::vector<int> out(kCount);
  for (size_t i = 0; i < kCount; i++) {
    in[i] = static_cast<int>(i);
    exp_out[i] = static_cast<int>(i);
  }

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  smirnov_i_radix_sort_simple_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(exp_out, out);
}
TEST(smirnov_i_radix_sort_simple_merge_seq, test_reverse_order) {
  constexpr size_t kCount = 100;

  // Create data
  std::vector<int> in(kCount, 0);
  std::vector<int> exp_out(kCount, 0);
  std::vector<int> out(kCount);
  for (size_t i = 0; i < kCount; i++) {
    in[kCount - 1 - i] = static_cast<int>(i);
  }
  for (size_t i = 0; i < kCount; i++) {
    exp_out[i] = static_cast<int>(i);
  }
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  smirnov_i_radix_sort_simple_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(exp_out, out);
}
TEST(smirnov_i_radix_sort_simple_merge_seq, test_771_elem) {
  constexpr size_t kCount = 771;

  // Create data
  std::vector<int> in(kCount, 0);
  std::vector<int> out(kCount);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, 1000);

  for (auto &num : in) {
    num = dist(gen);
  }
  std::vector<int> exp_out = in;
  std::ranges::sort(exp_out);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  smirnov_i_radix_sort_simple_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(exp_out, out);
}
