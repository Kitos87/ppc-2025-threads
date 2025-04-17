#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/sotskov_a_shell_sorting_with_simple_merging/include/ops_omp.hpp"

namespace sotskov_a_shell_sorting_with_simple_merging_omp {

std::vector<int> GenerateRandomVector(const RandomVectorParams &params) {
  std::random_device random_device;
  std::mt19937 generator(random_device());
  std::uniform_int_distribution<int> distribution(params.min_value, params.max_value);

  std::vector<int> random_vector(params.size);

  for (int &element : random_vector) {
    element = distribution(generator);
  }

  return random_vector;
}

void RunSortingTest(SortingTestParams &params, void (*sort_func)(std::vector<int> &)) {
  std::vector<int> out(params.input.size(), 0);

  sort_func(params.input);

  ASSERT_TRUE(std::ranges::is_sorted(params.input));

  std::shared_ptr<ppc::core::TaskData> task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(params.input.data()));
  task_data_omp->inputs_count.emplace_back(params.input.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  sotskov_a_shell_sorting_with_simple_merging_omp::TestTaskOpenMP test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.ValidationImpl(), true);
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();

  ASSERT_TRUE(std::ranges::is_sorted(out));
}

}  // namespace sotskov_a_shell_sorting_with_simple_merging_omp

TEST(sotskov_a_shell_sorting_with_simple_merging_omp, test_sort_positive_numbers) {
  sotskov_a_shell_sorting_with_simple_merging_omp::SortingTestParams params = {.expected = {1, 1, 2, 4, 5, 6, 7, 8},
                                                                               .input = {5, 1, 8, 6, 2, 7, 1, 4}};

  sotskov_a_shell_sorting_with_simple_merging_omp::RunSortingTest(
      params, sotskov_a_shell_sorting_with_simple_merging_omp::ShellSortWithSimpleMerging);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_omp, test_sort_negative_numbers) {
  sotskov_a_shell_sorting_with_simple_merging_omp::SortingTestParams params = {.expected = {-12, -10, -8, -7, -4, -3},
                                                                               .input = {-8, -3, -12, -7, -4, -10}};

  sotskov_a_shell_sorting_with_simple_merging_omp::RunSortingTest(
      params, sotskov_a_shell_sorting_with_simple_merging_omp::ShellSortWithSimpleMerging);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_omp, test_sort_ordered_array) {
  sotskov_a_shell_sorting_with_simple_merging_omp::SortingTestParams params = {.expected = {1, 2, 3, 4, 5, 6, 7, 8},
                                                                               .input = {1, 2, 3, 4, 5, 6, 7, 8}};

  sotskov_a_shell_sorting_with_simple_merging_omp::RunSortingTest(
      params, sotskov_a_shell_sorting_with_simple_merging_omp::ShellSortWithSimpleMerging);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_omp, test_sort_with_duplicates) {
  sotskov_a_shell_sorting_with_simple_merging_omp::SortingTestParams params = {.expected = {2, 2, 2, 4, 4, 6, 6, 8},
                                                                               .input = {4, 2, 2, 8, 4, 6, 6, 2}};

  sotskov_a_shell_sorting_with_simple_merging_omp::RunSortingTest(
      params, sotskov_a_shell_sorting_with_simple_merging_omp::ShellSortWithSimpleMerging);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_omp, test_sort_single_element) {
  sotskov_a_shell_sorting_with_simple_merging_omp::SortingTestParams params = {.expected = {77}, .input = {77}};

  sotskov_a_shell_sorting_with_simple_merging_omp::RunSortingTest(
      params, sotskov_a_shell_sorting_with_simple_merging_omp::ShellSortWithSimpleMerging);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_omp, test_sort_empty_array) {
  sotskov_a_shell_sorting_with_simple_merging_omp::SortingTestParams params = {.expected = {}, .input = {}};

  sotskov_a_shell_sorting_with_simple_merging_omp::RunSortingTest(
      params, sotskov_a_shell_sorting_with_simple_merging_omp::ShellSortWithSimpleMerging);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_omp, test_sort_random_vector) {
  sotskov_a_shell_sorting_with_simple_merging_omp::RandomVectorParams params = {
      .size = 20, .min_value = -100, .max_value = 100};
  std::vector<int> in = sotskov_a_shell_sorting_with_simple_merging_omp::GenerateRandomVector(params);
  std::vector<int> expected = in;

  std::ranges::sort(expected.begin(), expected.end());

  sotskov_a_shell_sorting_with_simple_merging_omp::SortingTestParams sorting_params = {.expected = expected,
                                                                                       .input = in};

  sotskov_a_shell_sorting_with_simple_merging_omp::RunSortingTest(
      sorting_params, sotskov_a_shell_sorting_with_simple_merging_omp::ShellSortWithSimpleMerging);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_omp, test_sort_reverse_ordered_array) {
  sotskov_a_shell_sorting_with_simple_merging_omp::SortingTestParams params = {.expected = {8, 7, 6, 5, 4, 3, 2, 1},
                                                                               .input = {8, 7, 6, 5, 4, 3, 2, 1}};

  sotskov_a_shell_sorting_with_simple_merging_omp::RunSortingTest(
      params, sotskov_a_shell_sorting_with_simple_merging_omp::ShellSortWithSimpleMerging);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_omp, test_sort_vector_with_identical_elements) {
  sotskov_a_shell_sorting_with_simple_merging_omp::SortingTestParams params = {.expected = {2, 2, 2, 2, 2, 2, 2, 2},
                                                                               .input = {2, 2, 2, 2, 2, 2, 2, 2}};

  sotskov_a_shell_sorting_with_simple_merging_omp::RunSortingTest(
      params, sotskov_a_shell_sorting_with_simple_merging_omp::ShellSortWithSimpleMerging);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_omp, test_sort_double_reverse) {
  sotskov_a_shell_sorting_with_simple_merging_omp::SortingTestParams params = {
      .expected = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5}, .input = {5, 4, 3, 2, 1, 5, 4, 3, 2, 1}};

  sotskov_a_shell_sorting_with_simple_merging_omp::RunSortingTest(
      params, sotskov_a_shell_sorting_with_simple_merging_omp::ShellSortWithSimpleMerging);
}
