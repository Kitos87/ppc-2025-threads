#include "seq/kalyakina_a_Shell_with_simple_merge/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

std::vector<unsigned int> kalyakina_a_shell_with_simple_merge_seq::ShellSortSequential::CalculationOfGapLengths(
    const unsigned int &size) {
  std::vector<unsigned int> result;
  unsigned int local_res = 1;
  for (unsigned int i = 1; (local_res * 3 <= size) || (local_res == 1); i++) {
    result.push_back(local_res);
    if (i % 2 != 0) {
      local_res = (unsigned int)((8 * pow(2, i)) - (6 * pow(2, (float)(i + 1) / 2)) + 1);
    } else {
      local_res = (unsigned int)((9 * pow(2, i)) - (9 * pow(2, (float)i / 2)) + 1);
    }
  }
  return result;
}

void kalyakina_a_shell_with_simple_merge_seq::ShellSortSequential::ShellSort(std::vector<int> &vec) {
  for (unsigned int k = Sedgwick_sequence_.size(); k > 0;) {
    unsigned int gap = Sedgwick_sequence_[--k];
    for (unsigned int i = 0; i < gap; i++) {
      for (unsigned int j = i; j < vec.size(); j += gap) {
        unsigned int index = j;
        int tmp = vec[index];
        while ((index >= gap) && (tmp < vec[index - gap])) {
          vec[index] = vec[index - gap];
          index -= gap;
        }
        vec[index] = tmp;
      }
    }
  }
}

bool kalyakina_a_shell_with_simple_merge_seq::ShellSortSequential::PreProcessingImpl() {
  input_ = std::vector<int>(task_data->inputs_count[0]);
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  std::ranges::copy(in_ptr, in_ptr + task_data->inputs_count[0], input_.begin());

  Sedgwick_sequence_ = CalculationOfGapLengths(input_.size());

  return true;
}

bool kalyakina_a_shell_with_simple_merge_seq::ShellSortSequential::ValidationImpl() {
  return (task_data->inputs_count[0] > 0) && (task_data->outputs_count[0] > 0) &&
         (task_data->inputs_count[0] == task_data->outputs_count[0]);
}

bool kalyakina_a_shell_with_simple_merge_seq::ShellSortSequential::RunImpl() {
  ShellSort(input_);
  return true;
}

bool kalyakina_a_shell_with_simple_merge_seq::ShellSortSequential::PostProcessingImpl() {
  for (size_t i = 0; i < input_.size(); i++) {
    reinterpret_cast<int *>(task_data->outputs[0])[i] = input_[i];
  }
  return true;
}
