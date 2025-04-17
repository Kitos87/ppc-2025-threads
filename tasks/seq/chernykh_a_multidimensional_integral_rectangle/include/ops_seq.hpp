#pragma once

#include <functional>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace chernykh_a_multidimensional_integral_rectangle_seq {

using Point = std::vector<double>;
using Function = std::function<double(const Point &)>;

class Dimension {
 public:
  explicit Dimension(double lower_bound, double upper_bound, int steps_count)
      : lower_bound_(lower_bound), upper_bound_(upper_bound), steps_count_(steps_count) {}
  [[nodiscard]] double GetLowerBound() const;
  [[nodiscard]] double GetUpperBound() const;
  [[nodiscard]] int GetStepsCount() const;
  [[nodiscard]] double GetStepSize() const;
  [[nodiscard]] bool IsValid() const;

 private:
  double lower_bound_{};
  double upper_bound_{};
  int steps_count_{};
};

class SequentialTask final : public ppc::core::Task {
 public:
  explicit SequentialTask(ppc::core::TaskDataPtr task_data, Function func)
      : Task(std::move(task_data)), func_(std::move(func)) {}
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  Function func_;
  std::vector<Dimension> dims_;
  double result_{};

  void FillPoint(int index, Point &point) const;
  [[nodiscard]] int GetTotalPoints() const;
  [[nodiscard]] double GetScalingFactor() const;
};

}  // namespace chernykh_a_multidimensional_integral_rectangle_seq
