#include "seq/belov_a_radix_sort_with_batcher_mergesort/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <vector>

namespace belov_a_radix_batcher_mergesort_seq {
int RadixBatcherMergesortSequential::GetNumberDigitCapacity(Bigint num) {
  return (num == 0 ? 1 : static_cast<int>(log10(std::fabs(num))) + 1);
}

void RadixBatcherMergesortSequential::Sort(std::vector<Bigint>& arr) {
  std::vector<Bigint> pos;
  std::vector<Bigint> neg;

  for (const auto& num : arr) {
    (num >= 0 ? pos : neg).push_back(std::abs(num));
  }

  RadixSort(pos, false);
  RadixSort(neg, true);

  size_t index = 0;

  for (const auto& num : neg) {
    arr[index++] = -num;
  }

  for (const auto& num : pos) {
    arr[index++] = num;
  }
}

void RadixBatcherMergesortSequential::RadixSort(std::vector<Bigint>& arr, bool invert) {
  if (arr.empty()) {
    return;
  }

  Bigint max_val = *std::ranges::max_element(arr);
  int max_val_digit_capacity = GetNumberDigitCapacity(max_val);
  int iter = 1;

  for (Bigint digit_place = 1; iter <= max_val_digit_capacity; digit_place *= 10, ++iter) {
    CountingSort(arr, digit_place);
  }

  if (invert) {
    std::ranges::reverse(arr);
  }
}

void RadixBatcherMergesortSequential::CountingSort(std::vector<Bigint>& arr, Bigint digit_place) {
  std::vector<Bigint> output(arr.size());
  int count[10] = {};

  for (const auto& num : arr) {
    Bigint index = (num / digit_place) % 10;
    count[index]++;
  }

  for (int i = 1; i < 10; i++) {
    count[i] += count[i - 1];
  }

  for (size_t i = arr.size() - 1; i < arr.size(); i--) {
    const Bigint& num = arr[i];
    Bigint index = (num / digit_place) % 10;
    output[--count[index]] = num;
  }

  std::ranges::copy(output, arr.begin());
}

bool RadixBatcherMergesortSequential::PreProcessingImpl() {
  n_ = task_data->inputs_count[0];
  auto* input_array_data = reinterpret_cast<Bigint*>(task_data->inputs[0]);
  array_.assign(input_array_data, input_array_data + n_);

  return true;
}

bool RadixBatcherMergesortSequential::ValidationImpl() {
  return (task_data->inputs.size() == 1 && !(task_data->inputs_count.size() < 2) && task_data->inputs_count[0] != 0 &&
          (task_data->inputs_count[0] == task_data->inputs_count[1]) && !task_data->outputs.empty());
}

bool RadixBatcherMergesortSequential::RunImpl() {
  Sort(array_);
  return true;
}

bool RadixBatcherMergesortSequential::PostProcessingImpl() {
  std::ranges::copy(array_, reinterpret_cast<Bigint*>(task_data->outputs[0]));
  return true;
}

}  // namespace belov_a_radix_batcher_mergesort_seq
