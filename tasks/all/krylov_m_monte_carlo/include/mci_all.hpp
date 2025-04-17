#pragma once

#include <boost/mpi/communicator.hpp>
#include <utility>

#include "./mci_common.hpp"
#include "core/task/include/task.hpp"

namespace krylov_m_monte_carlo {

class TaskALL : public TaskCommon {
 public:
  explicit TaskALL(ppc::core::TaskDataPtr task_data) : TaskCommon(std::move(task_data)) {}

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  boost::mpi::communicator world_;
};

}  // namespace krylov_m_monte_carlo