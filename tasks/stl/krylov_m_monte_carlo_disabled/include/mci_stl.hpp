#pragma once

#include <utility>

#include "./mci_common.hpp"
#include "core/task/include/task.hpp"

namespace krylov_m_monte_carlo {

class TaskSTL : public TaskCommon {
 public:
  explicit TaskSTL(ppc::core::TaskDataPtr task_data) : TaskCommon(std::move(task_data)) {}

  bool RunImpl() override;
};

}  // namespace krylov_m_monte_carlo