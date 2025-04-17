#pragma once

#include <utility>

#include "./mci_common.hpp"
#include "core/task/include/task.hpp"

namespace krylov_m_monte_carlo {

class TaskSequential : public TaskCommon {
 public:
  explicit TaskSequential(ppc::core::TaskDataPtr task_data) : TaskCommon(std::move(task_data)) {}

  bool RunImpl() override;
};

}  // namespace krylov_m_monte_carlo