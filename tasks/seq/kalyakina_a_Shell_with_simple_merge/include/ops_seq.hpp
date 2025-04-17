#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kalyakina_a_shell_with_simple_merge_seq {

class ShellSortSequential : public ppc::core::Task {
  static std::vector<unsigned int> CalculationOfGapLengths(const unsigned int& size);
  void ShellSort(std::vector<int>& vec);

 public:
  explicit ShellSortSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  std::vector<unsigned int> Sedgwick_sequence_;
};

}  // namespace kalyakina_a_shell_with_simple_merge_seq