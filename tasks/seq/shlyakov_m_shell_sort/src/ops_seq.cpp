#include "seq/shlyakov_m_shell_sort/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

bool shlyakov_m_shell_sort_seq::TestTaskSequential::PreProcessingImpl() {
  std::size_t input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  output_ = input_;

  return true;
}

bool shlyakov_m_shell_sort_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool shlyakov_m_shell_sort_seq::TestTaskSequential::RunImpl() {
  int n = static_cast<int>(input_.size());

  std::vector<int> gaps;
  for (int i = 1; i <= static_cast<int>(std::sqrt(n)) + 1; ++i) {
    int gap = static_cast<int>(n / std::pow(2.0, i));
    if (gap > 0) {
      gaps.push_back(gap);
    }
  }

  for (int k = static_cast<int>(gaps.size()) - 1; k >= 0; --k) {
    int gap = gaps[k];
    for (int start = 0; start < gap; ++start) {
      for (int i = start + gap; i < n; i += gap) {
        int key = output_[i];
        int j = i - gap;
        while (j >= start && output_[j] > key) {
          output_[j + gap] = output_[j];
          j -= gap;
        }
        output_[j + gap] = key;
      }
    }
  }
  return true;
}

bool shlyakov_m_shell_sort_seq::TestTaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); ++i) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}