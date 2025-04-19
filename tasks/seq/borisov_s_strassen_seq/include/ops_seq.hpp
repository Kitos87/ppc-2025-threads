#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace borisov_s_strassen_seq {

class SequentialStrassenSeq : public ppc::core::Task {
 public:
  explicit SequentialStrassenSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_;

  std::vector<double> output_;

  int rowsA_ = 0;
  int colsA_ = 0;
  int rowsB_ = 0;
  int colsB_ = 0;
};

}  // namespace borisov_s_strassen_seq
