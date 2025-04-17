#pragma once

#include <array>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

using Point = std::array<double, 2>;

namespace shvedova_v_graham_convex_hull_seq {

class GrahamConvexHullSequential : public ppc::core::Task {
 public:
  explicit GrahamConvexHullSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int points_count_{0};
  std::vector<Point> input_;
  std::vector<Point> res_;

  void PerformSort();
};

}  // namespace shvedova_v_graham_convex_hull_seq
