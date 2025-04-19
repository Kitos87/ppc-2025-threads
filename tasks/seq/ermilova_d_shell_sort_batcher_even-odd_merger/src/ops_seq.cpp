#include "seq/ermilova_d_shell_sort_batcher_even-odd_merger/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

namespace {
std::vector<int> SedgwickSequence(int n) {
  std::vector<int> gaps;
  int k = 0;
  while (true) {
    int gap = 0;
    if (k % 2 == 0) {
      gap = 9 * (1 << (2 * k)) - 9 * (1 << k) + 1;
    } else {
      gap = 8 * (1 << k) - 6 * (1 << ((k + 1) / 2)) + 1;
    }

    if (gap * 3 >= n) {
      break;
    }

    gaps.push_back(gap);
    k++;
  }
  return gaps;
}

void ShellSort(std::vector<int> &vec, const std::function<bool(int, int)> &comp) {
  int n = static_cast<int>(vec.size());
  std::vector<int> gaps = SedgwickSequence(n);

  for (int k = static_cast<int>(gaps.size()) - 1; k >= 0; k--) {
    int gap = gaps[k];
    for (int i = gap; i < n; i++) {
      int temp = vec[i];
      int j = 0;
      for (j = i; j >= gap && comp(vec[j - gap], temp); j -= gap) {
        vec[j] = vec[j - gap];
      }
      vec[j] = temp;
    }
  }
}
}  // namespace
bool ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential::PreProcessingImpl() {
  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  is_descending_ = *reinterpret_cast<bool *>(task_data->inputs[1]);
  input_.assign(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);

  return true;
}

bool ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential::ValidationImpl() {
  // Check equality of counts elements
  return (task_data->inputs_count[0] > 0) && (task_data->inputs_count[0] == task_data->outputs_count[0]);
}

bool ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential::RunImpl() {
  if (is_descending_) {
    ShellSort(input_, std::less());
  } else {
    ShellSort(input_, std::greater());
  }
  return true;
}

bool ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential::PostProcessingImpl() {
  auto *data = reinterpret_cast<int *>(task_data->outputs[0]);
  std::ranges::copy(input_, data);
  return true;
}
