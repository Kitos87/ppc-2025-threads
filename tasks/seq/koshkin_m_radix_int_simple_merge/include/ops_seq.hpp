#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace koshkin_m_radix_int_simple_merge {

class SeqT : public ppc::core::Task {
 public:
  explicit SeqT(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> in_, out_;
};

}  // namespace koshkin_m_radix_int_simple_merge