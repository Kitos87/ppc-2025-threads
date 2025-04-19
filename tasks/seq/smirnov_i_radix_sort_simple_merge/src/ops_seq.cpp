#include "seq/smirnov_i_radix_sort_simple_merge/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

bool smirnov_i_radix_sort_simple_merge_seq::TestTaskSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  mas_ = std::vector<int>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);

  return true;
}

bool smirnov_i_radix_sort_simple_merge_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool smirnov_i_radix_sort_simple_merge_seq::TestTaskSequential::RunImpl() {
  int longest = *std::ranges::max_element(mas_.begin(), mas_.end());
  int len = std::ceil(std::log10(longest + 1));
  std::vector<int> sorting(mas_.size());
  int base = 1;
  for (int j = 0; j < len; j++, base *= 10) {
    std::vector<int> counting(10, 0);
    for (size_t i = 0; i < mas_.size(); i++) {
      counting[mas_[i] / base % 10]++;
    }
    std::partial_sum(counting.begin(), counting.end(), counting.begin());
    for (int i = static_cast<int>(mas_.size() - 1); i >= 0; i--) {
      int pos = counting[mas_[i] / base % 10] - 1;
      sorting[pos] = mas_[i];
      counting[mas_[i] / base % 10]--;
    }
    std::swap(mas_, sorting);
  }
  output_ = mas_;
  return true;
}

bool smirnov_i_radix_sort_simple_merge_seq::TestTaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}