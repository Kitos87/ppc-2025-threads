#pragma once

#include <utility>

#include "core/task/include/task.hpp"
#include "ops_omp.hpp"

namespace korablev_v_sobel_edges_omp {

class TestTaskSeq : public ppc::core::Task {
 public:
  explicit TestTaskSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  Image in_;
  Image out_;
};

}  // namespace korablev_v_sobel_edges_omp