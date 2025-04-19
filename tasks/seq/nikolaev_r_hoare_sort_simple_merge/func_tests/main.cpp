#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "../include/ops_seq.hpp"
#include "core/task/include/task.hpp"

namespace {
std::vector<double> GenerateRandomVector(size_t len, double min_val = -1000.0, double max_val = 1000.0) {
  std::vector<double> vect(len);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(min_val, max_val);

  for (size_t i = 0; i < len; ++i) {
    vect[i] = dis(gen);
  }

  return vect;
}

class HoareSortTest : public testing::TestWithParam<size_t> {
 protected:
  static void CreateTest(size_t len) {
    std::vector<double> in = GenerateRandomVector(len);
    std::vector<double> out(len, 0.0);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_seq->inputs_count.emplace_back(in.size());
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_seq->outputs_count.emplace_back(out.size());

    nikolaev_r_hoare_sort_simple_merge_seq::HoareSortSimpleMergeSequential hoare_sort_simple_merge_sequential(
        task_data_seq);
    ASSERT_TRUE(hoare_sort_simple_merge_sequential.Validation());
    ASSERT_TRUE(hoare_sort_simple_merge_sequential.PreProcessing());
    ASSERT_TRUE(hoare_sort_simple_merge_sequential.Run());
    ASSERT_TRUE(hoare_sort_simple_merge_sequential.PostProcessing());

    std::vector<double> ref(len);
    std::ranges::copy(in, ref.begin());
    std::ranges::sort(ref);

    EXPECT_EQ(out, ref);
  }
};

TEST_P(HoareSortTest, sort_test) { CreateTest(GetParam()); }

INSTANTIATE_TEST_SUITE_P(nikolaev_r_hoare_sort_simple_merge_seq, HoareSortTest,
                         testing::Values(1, 2, 10, 100, 150, 200, 1000, 2000, 5000));

}  // namespace

TEST(nikolaev_r_hoare_sort_simple_merge_seq, test_empty_vect) {
  std::vector<double> in = {};
  std::vector<double> out = {};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs_count.emplace_back(out.size());

  nikolaev_r_hoare_sort_simple_merge_seq::HoareSortSimpleMergeSequential hoare_sort_simple_merge_sequential(
      task_data_seq);
  ASSERT_FALSE(hoare_sort_simple_merge_sequential.Validation());
}

TEST(nikolaev_r_hoare_sort_simple_merge_seq, test_reverse_order) {
  std::vector<double> in = {10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0};
  std::vector<double> out(in.size(), 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  nikolaev_r_hoare_sort_simple_merge_seq::HoareSortSimpleMergeSequential hoare_sort_simple_merge_sequential(
      task_data_seq);
  ASSERT_TRUE(hoare_sort_simple_merge_sequential.Validation());
  ASSERT_TRUE(hoare_sort_simple_merge_sequential.PreProcessing());
  ASSERT_TRUE(hoare_sort_simple_merge_sequential.Run());
  ASSERT_TRUE(hoare_sort_simple_merge_sequential.PostProcessing());

  std::vector<double> ref(in.size());
  std::ranges::copy(in, ref.begin());
  std::ranges::sort(ref);

  EXPECT_EQ(out, ref);
}

TEST(nikolaev_r_hoare_sort_simple_merge_seq, test_invalid_output) {
  std::vector<double> in = {1.0, 2.0, 3.0};
  std::vector<double> out = {};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  nikolaev_r_hoare_sort_simple_merge_seq::HoareSortSimpleMergeSequential hoare_sort_simple_merge_sequential(
      task_data_seq);
  ASSERT_FALSE(hoare_sort_simple_merge_sequential.Validation());
}

TEST(nikolaev_r_hoare_sort_simple_merge_seq, test_input_and_output_sizes_not_equal) {
  std::vector<double> in = {1.0, 2.0, 3.0};
  std::vector<double> out = {0.0, 0.0};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  nikolaev_r_hoare_sort_simple_merge_seq::HoareSortSimpleMergeSequential hoare_sort_simple_merge_sequential(
      task_data_seq);
  ASSERT_FALSE(hoare_sort_simple_merge_sequential.Validation());
}