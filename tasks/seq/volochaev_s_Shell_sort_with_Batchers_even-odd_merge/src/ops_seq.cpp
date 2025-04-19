#include "seq/volochaev_s_Shell_sort_with_Batchers_even-odd_merge/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential::PreProcessingImpl() {
  // Init value for input and output
  unsigned int size = task_data->inputs_count[0];
  auto* input_pointer = reinterpret_cast<int*>(task_data->inputs[0]);
  array_ = std::vector<int>(input_pointer, input_pointer + size);

  return true;
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs_count[0] > 0 && task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential::RunImpl() {
  ShellSort(array_);

  return true;
}

void volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential::ShellSort(std::vector<int>& arr) {
  int n = static_cast<int>(arr.size());
  int gap = n / 2;

  while (gap > 0) {
    for (int i = gap; i < n; ++i) {
      int temp = arr[i];
      int j = i;
      while (j >= gap && arr[j - gap] > temp) {
        arr[j] = arr[j - gap];
        j -= gap;
      }
      arr[j] = temp;
    }
    gap /= 2;
  }
}

bool volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential::PostProcessingImpl() {
  for (size_t i = 0; i < array_.size(); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = array_[i];
  }
  return true;
}
