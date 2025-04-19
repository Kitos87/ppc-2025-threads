#include "seq/sotskov_a_shell_sorting_with_simple_merging/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

void sotskov_a_shell_sorting_with_simple_merging_seq::ShellSortWithSimpleMerging(std::vector<int>& arr) {
  int array_size = static_cast<int>(arr.size());

  int gap = 1;
  while (gap < array_size / 3) {
    gap = gap * 3 + 1;
  }

  while (gap > 0) {
    for (int i = gap; i < array_size; ++i) {
      int current_element = arr[i];
      int j = i;

      while (j >= gap && arr[j - gap] > current_element) {
        arr[j] = arr[j - gap];
        j -= gap;
      }
      arr[j] = current_element;
    }
    gap /= 3;
  }
}

bool sotskov_a_shell_sorting_with_simple_merging_seq::TestTaskSequential::PreProcessingImpl() {
  input_ = std::vector<int>(task_data->inputs_count[0]);
  auto* temp_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  std::ranges::copy(temp_ptr, temp_ptr + task_data->inputs_count[0], input_.begin());

  return true;
}

bool sotskov_a_shell_sorting_with_simple_merging_seq::TestTaskSequential::ValidationImpl() {
  std::size_t input_size = task_data->inputs_count[0];
  std::size_t output_size = task_data->outputs_count[0];

  return (input_size == output_size);
}

bool sotskov_a_shell_sorting_with_simple_merging_seq::TestTaskSequential::RunImpl() {
  ShellSortWithSimpleMerging(input_);
  return true;
}

bool sotskov_a_shell_sorting_with_simple_merging_seq::TestTaskSequential::PostProcessingImpl() {
  int* output = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(input_.begin(), input_.end(), output);
  return true;
}
