#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "all/korovin_n_qsort_batcher/include/ops_all.hpp"
#include "core/task/include/task.hpp"

namespace {
struct GenParams {
  int size;
  int lower_bound = -500;
  int upper_bound = 500;
};

std::vector<int> GenerateRndVector(GenParams param) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(param.lower_bound, param.upper_bound);

  std::vector<int> v1(param.size);
  for (auto &num : v1) {
    num = dist(gen);
  }
  return v1;
}

void RunTest(std::vector<int> &in) {
  std::vector<int> out(in.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(in.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  korovin_n_qsort_batcher_all::TestTaskALL test_task_all(task_data_all);
  ASSERT_TRUE(test_task_all.Validation());
  ASSERT_TRUE(test_task_all.PreProcessing());
  ASSERT_TRUE(test_task_all.Run());
  ASSERT_TRUE(test_task_all.PostProcessing());
  ASSERT_TRUE(std::ranges::is_sorted(out));
}
}  // namespace

TEST(korovin_n_qsort_batcher_all, test_unsort) {
  std::vector<int> in = {11, 5, 1, 42, 3, 8, 7, 25, 6};
  RunTest(in);
}

TEST(korovin_n_qsort_batcher_all, test_reverse_sort) {
  std::vector<int> in = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  RunTest(in);
}

TEST(korovin_n_qsort_batcher_all, test_double_reverse_sort) {
  std::vector<int> in = {5, 4, 3, 2, 1, 5, 4, 3, 2, 1};
  RunTest(in);
}

TEST(korovin_n_qsort_batcher_all, test_repeating_25_87) {
  std::vector<int> in = {25, 87, 25, 87, 1, 25, 87, 0, -5};
  RunTest(in);
}

TEST(korovin_n_qsort_batcher_all, test_all_elements_equal_25) {
  std::vector<int> in(50, 25);
  RunTest(in);
}

TEST(korovin_n_qsort_batcher_all, test_extreme_values_with_25_87) {
  std::vector<int> in = {INT_MIN, 25, INT_MAX, 87, INT_MIN, 25, 0, INT_MAX, 87};
  RunTest(in);
}

TEST(korovin_n_qsort_batcher_all, test_empty_sort) {
  std::vector<int> in = {};
  RunTest(in);
}

TEST(korovin_n_qsort_batcher_all, test_one_el_sort) {
  std::vector<int> in = {42};
  RunTest(in);
}

TEST(korovin_n_qsort_batcher_all, test_already_sort) {
  std::vector<int> in = {1, 2, 3, 4, 5, 6};
  RunTest(in);
}

TEST(korovin_n_qsort_batcher_all, test_random_sort) {
  auto param = GenParams();
  param.size = 20;

  std::vector<int> in = GenerateRndVector(param);
  RunTest(in);
}

TEST(korovin_n_qsort_batcher_all, test_random_sort_100) {
  auto param = GenParams();
  param.size = 100;

  std::vector<int> in = GenerateRndVector(param);
  RunTest(in);
}

TEST(korovin_n_qsort_batcher_all, test_random_sort_500) {
  auto param = GenParams();
  param.size = 500;

  std::vector<int> in = GenerateRndVector(param);
  RunTest(in);
}

TEST(korovin_n_qsort_batcher_all, test_random_sort_1000) {
  auto param = GenParams();
  param.size = 1000;

  std::vector<int> in = GenerateRndVector(param);
  RunTest(in);
}
