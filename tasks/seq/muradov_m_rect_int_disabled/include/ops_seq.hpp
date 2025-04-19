#pragma once

#include <functional>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace muradov_m_rect_int_seq {

using Matfun = std::function<double(const std::vector<double>&)>;
using FunArgs = std::vector<double>;

class RectIntTaskSequential : public ppc::core::Task {
 public:
  explicit RectIntTaskSequential(ppc::core::TaskDataPtr task_data, Matfun fun)
      : Task(std::move(task_data)), fun_(std::move(fun)) {}

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  Matfun fun_;
  int grains_;
  std::vector<std::pair<double, double>> bounds_;
  double res_;
};

}  // namespace muradov_m_rect_int_seq