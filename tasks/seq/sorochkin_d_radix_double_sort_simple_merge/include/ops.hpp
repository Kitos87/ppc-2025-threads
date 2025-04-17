#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sorochkin_d_radix_double_sort_simple_merge_seq {

class SortTask : public ppc::core::Task {
 public:
  explicit SortTask(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_, output_;
};

}  // namespace sorochkin_d_radix_double_sort_simple_merge_seq