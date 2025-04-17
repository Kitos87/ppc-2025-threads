#pragma once
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sharamygina_i_multi_dim_monte_carlo_seq {
class MultiDimMonteCarloTask : public ppc::core::Task {
 public:
  explicit MultiDimMonteCarloTask(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> boundaries_;
  int number_of_iterations_ = 0;
  double final_result_ = 0.0;
  std::function<double(const std::vector<double>&)> integrating_function_;
};
}  // namespace sharamygina_i_multi_dim_monte_carlo_seq