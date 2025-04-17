#include "seq/gusev_n_sorting_int_simple_merging/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

void gusev_n_sorting_int_simple_merging_seq::TestTaskSequential::RadixSort(std::vector<int>& arr) {
  if (arr.empty()) {
    return;
  }

  std::vector<int> negatives;
  std::vector<int> positives;

  for (int num : arr) {
    if (num < 0) {
      negatives.push_back(-num);
    } else {
      positives.push_back(num);
    }
  }

  if (!negatives.empty()) {
    RadixSortForNonNegative(negatives);
    std::ranges::reverse(negatives.begin(), negatives.end());
    std::ranges::for_each(negatives, [](int& num) { num = -num; });
  }

  if (!positives.empty()) {
    RadixSortForNonNegative(positives);
  }

  arr.clear();
  arr.insert(arr.end(), negatives.begin(), negatives.end());
  arr.insert(arr.end(), positives.begin(), positives.end());
}

void gusev_n_sorting_int_simple_merging_seq::TestTaskSequential::RadixSortForNonNegative(std::vector<int>& arr) {
  if (arr.empty()) {
    return;
  }

  int max = *std::ranges::max_element(arr.begin(), arr.end());
  for (int exp = 1; max / exp > 0; exp *= 10) {
    CountingSort(arr, exp);
  }
}

void gusev_n_sorting_int_simple_merging_seq::TestTaskSequential::CountingSort(std::vector<int>& arr, int exp) {
  std::vector<int> output(arr.size());
  std::vector<int> count(10, 0);

  for (int num : arr) {
    int digit = (num / exp) % 10;
    count[digit]++;
  }

  std::partial_sum(count.begin(), count.end(), count.begin());

  for (std::vector<int>::size_type i = arr.size(); i > 0; i--) {
    int digit = (arr[i - 1] / exp) % 10;
    output[count[digit] - 1] = arr[i - 1];
    count[digit]--;
  }

  arr = output;
}

bool gusev_n_sorting_int_simple_merging_seq::TestTaskSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);
  output_ = std::vector<int>(input_size);

  return true;
}

bool gusev_n_sorting_int_simple_merging_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool gusev_n_sorting_int_simple_merging_seq::TestTaskSequential::RunImpl() {
  gusev_n_sorting_int_simple_merging_seq::TestTaskSequential::RadixSort(input_);
  return true;
}

bool gusev_n_sorting_int_simple_merging_seq::TestTaskSequential::PostProcessingImpl() {
  std::ranges::copy(input_, reinterpret_cast<int*>(task_data->outputs[0]));
  return true;
}
