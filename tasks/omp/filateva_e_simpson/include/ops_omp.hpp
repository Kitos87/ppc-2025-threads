#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace filateva_e_simpson_omp {

using Func = double (*)(std::vector<double>);

class Simpson : public ppc::core::Task {
 public:
  explicit Simpson(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  size_t mer_;
  std::vector<double> a_, b_;
  size_t steps_{};
  double res_{};

  Func f_;
};
}  // namespace filateva_e_simpson_omp