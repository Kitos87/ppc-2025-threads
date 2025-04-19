#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace nikolaev_r_hoare_sort_simple_merge_seq {

class HoareSortSimpleMergeSequential : public ppc::core::Task {
 public:
  explicit HoareSortSimpleMergeSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void QuickSort(size_t low, size_t high);

 private:
  std::vector<double> vect_;
  size_t vect_size_{};

  size_t Partition(size_t low, size_t high);
};

}  // namespace nikolaev_r_hoare_sort_simple_merge_seq