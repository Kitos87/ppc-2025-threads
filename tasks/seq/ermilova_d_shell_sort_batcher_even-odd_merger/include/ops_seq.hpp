#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace ermilova_d_shell_sort_batcher_even_odd_merger_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, output_;
  bool is_descending_;
};

}  // namespace ermilova_d_shell_sort_batcher_even_odd_merger_seq