#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/tsatsyn_a_radix_sort_simple_merge_seq/include/ops_seq.hpp"
std::vector<double> tsatsyn_a_radix_sort_simple_merge_seq::GetRandomVector(int sz, int a, int b) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dis(a, b);
  std::vector<double> vec(sz);
  for (int i = 0; i < sz; ++i) {
    vec[i] = dis(gen);
  }
  return vec;
}
TEST(tsatsyn_a_radix_sort_simple_merge_seq, reverse_order_100) {
  // Create data
  int arrsize = 100;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);

  in = {100.71829, 99.46283, 98.35147, 97.92461, 96.58372, 95.29164, 94.73918, 93.15627, 92.84539, 91.37261,
        90.12845,  89.65432, 88.29173, 87.45682, 86.91837, 85.37261, 84.65429, 83.29174, 82.91837, 81.45682,
        80.37261,  79.65429, 78.29174, 77.91837, 76.45682, 75.37261, 74.65429, 73.29174, 72.91837, 71.45682,
        70.37261,  69.65429, 68.29174, 67.91837, 66.45682, 65.37261, 64.65429, 63.29174, 62.91837, 61.45682,
        60.37261,  59.65429, 58.29174, 57.91837, 56.45682, 55.37261, 54.65429, 53.29174, 52.91837, 51.45682,
        50.37261,  49.65429, 48.29174, 47.91837, 46.45682, 45.37261, 44.65429, 43.29174, 42.91837, 41.45682,
        40.37261,  39.65429, 38.29174, 37.91837, 36.45682, 35.37261, 34.65429, 33.29174, 32.91837, 31.45682,
        30.37261,  29.65429, 28.29174, 27.91837, 26.45682, 25.37261, 24.65429, 23.29174, 22.91837, 21.45682,
        20.37261,  19.65429, 18.29174, 17.91837, 16.45682, 15.37261, 14.65429, 13.29174, 12.91837, 11.45682,
        -10.37261, -9.65429, -8.29174, -7.91837, -6.45682, -5.37261, -4.65429, -3.29174, -2.91837, -1.45682};
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  // Create Task
  tsatsyn_a_radix_sort_simple_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_seq, negative_double_100) {
  // Create data
  int arrsize = 100;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);

  in = tsatsyn_a_radix_sort_simple_merge_seq::GetRandomVector(arrsize, -100, 100);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  // Create Task
  tsatsyn_a_radix_sort_simple_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_seq, negative_double_1000) {
  // Create data
  int arrsize = 1000;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);

  in = tsatsyn_a_radix_sort_simple_merge_seq::GetRandomVector(arrsize, -100, 100);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  // Create Task
  tsatsyn_a_radix_sort_simple_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_seq, negative_double_10000) {
  // Create data
  int arrsize = 10000;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);

  in = tsatsyn_a_radix_sort_simple_merge_seq::GetRandomVector(arrsize, -100, 100);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  // Create Task
  tsatsyn_a_radix_sort_simple_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_seq, negative_double_100000) {
  // Create data
  int arrsize = 100000;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);

  in = tsatsyn_a_radix_sort_simple_merge_seq::GetRandomVector(arrsize, -100, 100);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  // Create Task
  tsatsyn_a_radix_sort_simple_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_seq, pozitive_double_100) {
  // Create data
  int arrsize = 100;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);

  in = tsatsyn_a_radix_sort_simple_merge_seq::GetRandomVector(arrsize, 0, 100);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  // Create Task
  tsatsyn_a_radix_sort_simple_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_seq, pozitive_double_1000) {
  // Create data
  int arrsize = 1000;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);

  in = tsatsyn_a_radix_sort_simple_merge_seq::GetRandomVector(arrsize, 0, 100);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  // Create Task
  tsatsyn_a_radix_sort_simple_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_seq, pozitive_double_10000) {
  // Create data
  int arrsize = 10000;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);

  in = tsatsyn_a_radix_sort_simple_merge_seq::GetRandomVector(arrsize, 0, 100);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  // Create Task
  tsatsyn_a_radix_sort_simple_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}
TEST(tsatsyn_a_radix_sort_simple_merge_seq, pozitive_double_100000) {
  // Create data
  int arrsize = 100000;
  std::vector<double> in;
  std::vector<double> out(arrsize, 0);

  in = tsatsyn_a_radix_sort_simple_merge_seq::GetRandomVector(arrsize, 0, 100);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  // Create Task
  tsatsyn_a_radix_sort_simple_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}