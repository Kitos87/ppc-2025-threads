#include "omp/kalyakina_a_Shell_with_simple_merge/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

std::vector<unsigned int> kalyakina_a_shell_with_simple_merge_omp::ShellSortOpenMP::CalculationOfGapLengths(
    unsigned int size) {
  std::vector<unsigned int> result;
  unsigned int local_res = 1;
  for (unsigned int i = 1; (local_res * 3 <= size) || (local_res == 1); i++) {
    result.push_back(local_res);
    if (i % 2 != 0) {
      local_res = static_cast<unsigned int>((8 * pow(2, i)) - (6 * pow(2, static_cast<float>(i + 1) / 2)) + 1);
    } else {
      local_res = static_cast<unsigned int>((9 * pow(2, i)) - (9 * pow(2, static_cast<float>(i) / 2)) + 1);
    }
  }
  return result;
}

void kalyakina_a_shell_with_simple_merge_omp::ShellSortOpenMP::ShellSort(unsigned int left, unsigned int right) {
  for (unsigned int k = Sedgwick_sequence_.size(); k > 0;) {
    unsigned int gap = Sedgwick_sequence_[--k];
    for (unsigned int i = left; i < left + gap; i++) {
      for (unsigned int j = i; j < right; j += gap) {
        unsigned int index = j;
        int tmp = output_[index];
        while ((index >= i + gap) && (tmp < output_[index - gap])) {
          output_[index] = output_[index - gap];
          index -= gap;
        }
        output_[index] = tmp;
      }
    }
  }
}

void kalyakina_a_shell_with_simple_merge_omp::ShellSortOpenMP::SimpleMergeSort(unsigned int left, unsigned int middle,
                                                                               unsigned int right) {
  std::vector<int> first_part(middle - left);
  std::copy(output_.begin() + left, output_.begin() + middle, first_part.begin());
  unsigned int l = 0;
  unsigned int r = middle;
  unsigned int j = left;
  for (; (l < first_part.size()) && (r < right); j++) {
    if (first_part[l] < output_[r]) {
      output_[j] = first_part[l++];
    } else {
      output_[j] = output_[r++];
    }
  }
  while (l < first_part.size()) {
    output_[j++] = first_part[l++];
  }
  while (r < right) {
    output_[j++] = output_[r++];
  }
}

bool kalyakina_a_shell_with_simple_merge_omp::ShellSortOpenMP::PreProcessingImpl() {
  input_ = std::vector<int>(task_data->inputs_count[0]);
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  std::copy(in_ptr, in_ptr + task_data->inputs_count[0], input_.begin());

  output_ = std::vector<int>(task_data->inputs_count[0]);
  std::ranges::copy(input_, output_.begin());

  return true;
}

bool kalyakina_a_shell_with_simple_merge_omp::ShellSortOpenMP::ValidationImpl() {
  return (task_data->inputs_count[0] > 0) && (task_data->outputs_count[0] > 0) &&
         (task_data->inputs_count[0] == task_data->outputs_count[0]);
}

bool kalyakina_a_shell_with_simple_merge_omp::ShellSortOpenMP::RunImpl() {
  std::vector<std::pair<unsigned int, unsigned int>> bounds;
  unsigned int num = (static_cast<unsigned int>(omp_get_max_threads()) > output_.size())
                         ? output_.size()
                         : (unsigned int)omp_get_max_threads();
  unsigned int part = output_.size() / num;
  unsigned int reminder = output_.size() % num;
  unsigned int left = 0;
  Sedgwick_sequence_ = CalculationOfGapLengths(output_.size() / num);
  for (unsigned int i = 0; i < num; i++) {
    unsigned int right = (i < reminder) ? left + part + 1 : left + part;
    bounds.emplace_back(left, right);
    left = right;
  }
#pragma omp parallel for schedule(static)
  for (int i = 0; i < static_cast<int>(num); i++) {
    ShellSort(bounds[i].first, bounds[i].second);
  }
  num = std::ceil(static_cast<double>(num) / 2);
  unsigned int step = 1;
  while (step < bounds.size()) {
    step *= 2;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(num); i++) {
      unsigned int middle = (step / 2) + (step * i);
      if (middle < bounds.size()) {
        SimpleMergeSort(
            bounds[i * step].first, bounds[middle].first,
            bounds[(bounds.size() - 1 < (i + 1) * step - 1) ? bounds.size() - 1 : ((i + 1) * step) - 1].second);
      }
    }
    num = std::ceil(static_cast<double>(num) / 2);
  }
  return true;
}

bool kalyakina_a_shell_with_simple_merge_omp::ShellSortOpenMP::PostProcessingImpl() {
  std::ranges::copy(output_, reinterpret_cast<int *>(task_data->outputs[0]));
  return true;
}
