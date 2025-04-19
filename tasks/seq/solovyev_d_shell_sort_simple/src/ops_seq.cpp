#include "seq/solovyev_d_shell_sort_simple/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

bool solovyev_d_shell_sort_simple_seq::TaskSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);
  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);

  return true;
}

bool solovyev_d_shell_sort_simple_seq::TaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool solovyev_d_shell_sort_simple_seq::TaskSequential::RunImpl() {
  unsigned int gap = input_.size() / 2;
  while (gap > 0) {
    for (size_t i = gap; i < input_.size(); i++) {
      int val = input_[i];
      size_t j = i;
      while (j >= gap && input_[j - gap] > val) {
        input_[j] = input_[j - gap];
        j -= gap;
      }
      input_[j] = val;
    }
    gap = gap / 2;
  }
  return true;
}

bool solovyev_d_shell_sort_simple_seq::TaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int *>(task_data->outputs[0])[i] = input_[i];
  }
  return true;
}
