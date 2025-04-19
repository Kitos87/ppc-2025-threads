#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/kalyakina_a_Shell_with_simple_merge/include/ops_seq.hpp"

namespace {

std::vector<int> CreateReverseSortedVector(unsigned int size, int left);
std::vector<int> CreateRandomVector(unsigned int size, int left, int right);
void TestOfFunction(std::vector<int>& in);

std::vector<int> CreateReverseSortedVector(unsigned int size, const int left) {
  std::vector<int> result;
  while (size-- != 0) {
    result.push_back(left + (int)size);
  }
  return result;
}

std::vector<int> CreateRandomVector(unsigned int size, const int left, const int right) {
  std::vector<int> result;
  std::random_device dev;
  std::mt19937 gen(dev());
  while (size-- != 0) {
    result.push_back((int)(gen() % (int)(right - left)) + left);
  }
  return result;
}

void TestOfFunction(std::vector<int>& in) {
  std::vector<int> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_sequential = std::make_shared<ppc::core::TaskData>();

  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_sequential->inputs_count.emplace_back(in.size());
  task_data_sequential->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_sequential->outputs_count.emplace_back(out.size());

  kalyakina_a_shell_with_simple_merge_seq::ShellSortSequential task_sequential(task_data_sequential);

  ASSERT_EQ(task_sequential.Validation(), true);
  task_sequential.PreProcessing();
  task_sequential.Run();
  task_sequential.PostProcessing();

  ASSERT_TRUE(std::ranges::is_sorted(out.begin(), out.end()));
}
}  // namespace

TEST(kalyakina_a_Shell_with_simple_merge_seq, test_of_Validation1) {
  std::vector<int> in = {2, 9, 5, 1, 4, 8, 3};
  std::vector<int> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_sequential = std::make_shared<ppc::core::TaskData>();

  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_sequential->inputs_count.emplace_back(0);
  task_data_sequential->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_sequential->outputs_count.emplace_back(out.size());

  kalyakina_a_shell_with_simple_merge_seq::ShellSortSequential task_sequential(task_data_sequential);

  ASSERT_EQ(task_sequential.Validation(), false);
}

TEST(kalyakina_a_Shell_with_simple_merge_seq, test_of_Validation2) {
  std::vector<int> in = {2, 9, 5, 1, 4, 8, 3};
  std::vector<int> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_sequential = std::make_shared<ppc::core::TaskData>();

  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_sequential->inputs_count.emplace_back(in.size());
  task_data_sequential->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_sequential->outputs_count.emplace_back(0);

  kalyakina_a_shell_with_simple_merge_seq::ShellSortSequential task_sequential(task_data_sequential);

  ASSERT_EQ(task_sequential.Validation(), false);
}

TEST(kalyakina_a_Shell_with_simple_merge_seq, test_of_Validation3) {
  std::vector<int> in = {2, 9, 5, 1, 4, 8, 3};
  std::vector<int> out(in.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_sequential = std::make_shared<ppc::core::TaskData>();

  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_sequential->inputs_count.emplace_back(in.size());
  task_data_sequential->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_sequential->outputs_count.emplace_back(out.size() + 1);

  kalyakina_a_shell_with_simple_merge_seq::ShellSortSequential task_sequential(task_data_sequential);

  ASSERT_EQ(task_sequential.Validation(), false);
}

TEST(kalyakina_a_Shell_with_simple_merge_seq, small_fixed_vector) {
  std::vector<int> in = {2, 9, 5, 1, 4, 8, 3, 2};
  TestOfFunction(in);
}

TEST(kalyakina_a_Shell_with_simple_merge_seq, medium_fixed_vector) {
  std::vector<int> in = {2,  9,  5,  1,  4,  8,  3,  11, 34, 12, 6,  29,  13,   7,     56,     32,      88,
                         90, 94, 78, 54, 47, 37, 77, 22, 44, 55, 66, 123, 1234, 12345, 123456, 1234567, 0};
  TestOfFunction(in);
}

TEST(kalyakina_a_Shell_with_simple_merge_seq, reverse_sorted_vector_50) {
  std::vector<int> in = CreateReverseSortedVector(50, -10);
  TestOfFunction(in);
}

TEST(kalyakina_a_Shell_with_simple_merge_seq, reverse_sorted_vector_100) {
  std::vector<int> in = CreateReverseSortedVector(100, -10);
  TestOfFunction(in);
}

TEST(kalyakina_a_Shell_with_simple_merge_seq, reverse_sorted_vector_1000) {
  std::vector<int> in = CreateReverseSortedVector(1000, -10);
  TestOfFunction(in);
}

TEST(kalyakina_a_Shell_with_simple_merge_seq, reverse_sorted_vector_10000) {
  std::vector<int> in = CreateReverseSortedVector(10000, -10);
  TestOfFunction(in);
}

TEST(kalyakina_a_Shall_with_simple_merge_seq, random_vector_50) {
  std::vector<int> in = CreateRandomVector(50, -7000, 7000);
  TestOfFunction(in);
}

TEST(kalyakina_a_Shall_with_simple_merge_seq, random_vector_100) {
  std::vector<int> in = CreateRandomVector(100, -7000, 7000);
  TestOfFunction(in);
}

TEST(kalyakina_a_Shall_with_simple_merge_seq, random_vector_1000) {
  std::vector<int> in = CreateRandomVector(1000, -7000, 7000);
  TestOfFunction(in);
}

TEST(kalyakina_a_Shall_with_simple_merge_seq, random_vector_10000) {
  std::vector<int> in = CreateRandomVector(10000, -7000, 7000);
  TestOfFunction(in);
}
