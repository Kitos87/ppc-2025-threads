#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace malyshev_v_radix_sort_seq {

class RadixSortSequential : public ppc::core::Task {
 public:
  explicit RadixSortSequential(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> res_;
};
}  // namespace malyshev_v_radix_sort_seq