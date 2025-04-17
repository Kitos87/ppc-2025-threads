#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace gromov_a_fox_algorithm_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> A_, B_, output_;
  int n_;
  int block_size_;
};

}  // namespace gromov_a_fox_algorithm_seq