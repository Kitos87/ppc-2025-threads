#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace poroshin_v_multi_integral_with_trapez_method_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> &task_data,
                              std::function<double(std::vector<double> &args)> func)
      : Task(task_data), dim_(task_data->inputs_count[0]), func_(std::move(func)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  void CountMultiIntegralTrapezMethodSeq();
  std::vector<std::pair<double, double>> limits_;
  size_t dim_;
  std::function<double(std::vector<double> &args)> func_;
  std::vector<int> n_;
  double res_{};
};

}  // namespace poroshin_v_multi_integral_with_trapez_method_seq