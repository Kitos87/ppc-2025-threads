#pragma once

#include <utility>
#include <vector>

#include "chc.hpp"
#include "core/task/include/task.hpp"

namespace voroshilov_v_convex_hull_components_seq {

class ChcTaskSequential : public ppc::core::Task {
 public:
  explicit ChcTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  Image imageIn_;
  std::vector<Hull> hullsOut_;
};

}  // namespace voroshilov_v_convex_hull_components_seq
