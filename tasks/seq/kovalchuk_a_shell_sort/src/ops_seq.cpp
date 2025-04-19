#include "seq/kovalchuk_a_shell_sort/include/ops_seq.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kovalchuk_a_shell_sort {

ShellSortSequential::ShellSortSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

bool ShellSortSequential::PreProcessingImpl() {
  auto* input_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_.assign(input_ptr, input_ptr + task_data->inputs_count[0]);
  return true;
}

bool ShellSortSequential::ValidationImpl() {
  return !task_data->inputs_count.empty() && !task_data->outputs_count.empty() &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool ShellSortSequential::RunImpl() {
  ShellSort();
  return true;
}

void ShellSortSequential::ShellSort() {
  if (input_.empty()) {
    return;
  }

  int n = static_cast<int>(input_.size());
  for (int gap = n / 2; gap > 0; gap /= 2) {
    for (int i = gap; i < n; ++i) {
      int temp = input_[i];
      int j = i;
      for (; j >= gap && input_[j - gap] > temp; j -= gap) {
        input_[j] = input_[j - gap];
      }
      input_[j] = temp;
    }
  }
}

bool ShellSortSequential::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(input_, output_ptr);
  return true;
}

}  // namespace kovalchuk_a_shell_sort