#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace korovin_n_qsort_batcher_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  static int GetRandomIndex(int low, int high);
  static void QuickSort(std::vector<int>& arr, int low, int high);
};

}  // namespace korovin_n_qsort_batcher_seq