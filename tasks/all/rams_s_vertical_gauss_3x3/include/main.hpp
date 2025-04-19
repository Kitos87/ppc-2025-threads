#pragma once

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace rams_s_vertical_gauss_3x3_all {

class TaskAll : public ppc::core::Task {
 public:
  explicit TaskAll(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  uint32_t height_, width_;
  std::vector<uint8_t> input_, output_;
  std::vector<float> kernel_;
  boost::mpi::communicator world_;
};

}  // namespace rams_s_vertical_gauss_3x3_all
