#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sotskov_a_shell_sorting_with_simple_merging_omp {
struct RandomVectorParams {
  int size;
  int min_value;
  int max_value;
};
struct SortingTestParams {
  std::vector<int> expected;
  std::vector<int> input;
};
void RunSortingTest(SortingTestParams& params, void (*sort_func)(std::vector<int>&));
void ShellSortWithSimpleMerging(std::vector<int>& arr);
void ShellSort(std::vector<int>& arr, int left, int right);
void ParallelMerge(std::vector<int>& arr, int left, int mid, int right, std::vector<int>& temp_buffer);
std::vector<int> GenerateRandomVector(const RandomVectorParams& params);
class TestTaskOpenMP : public ppc::core::Task {
 public:
  explicit TestTaskOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, result_;
};
}  // namespace sotskov_a_shell_sorting_with_simple_merging_omp