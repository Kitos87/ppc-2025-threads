#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vladimirova_j_m_monte_karlo_seq {

struct BoundariesIntegral {
  double min;
  double max;
};

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_, output_;
  bool (*func_)(std::vector<double>, size_t);
  std::vector<BoundariesIntegral> var_integr_;
  size_t var_size_{};
  size_t accuracy_;
};

}  // namespace vladimirova_j_m_monte_karlo_seq
