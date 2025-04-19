#include "seq/malyshev_v_radix_sort/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace malyshev_v_radix_sort_seq {
namespace {
union DoubleWrapper {
  double d;
  uint64_t u;
};

void CountingSort(std::vector<double>& arr, int exp) {
  const int radix = 256;
  size_t n = arr.size();
  std::vector<double> output(n);
  std::vector<int> count(radix, 0);

  for (size_t i = 0; i < n; i++) {
    DoubleWrapper dw;
    dw.d = arr[i];
    uint64_t value = dw.u;
    auto index = static_cast<int>((value >> (8 * exp)) & 0xFF);
    count[index]++;
  }

  for (int i = 1; i < radix; i++) {
    count[i] += count[i - 1];
  }

  for (int i = static_cast<int>(n) - 1; i >= 0; i--) {
    DoubleWrapper dw;
    dw.d = arr[i];
    uint64_t value = dw.u;
    auto index = static_cast<int>((value >> (8 * exp)) & 0xFF);
    output[count[index] - 1] = arr[i];
    count[index]--;
  }

  arr = output;
}

void RadixSort(std::vector<double>& arr) {
  if (arr.empty()) {
    return;
  }

  auto middle = std::ranges::partition(arr, [](double x) { return x < 0.0; });

  std::vector<double> neg_part(arr.begin(), middle.begin());
  for (auto& num : neg_part) {
    DoubleWrapper dw;
    dw.d = num;
    dw.u ^= 0x8000000000000000;
    num = dw.d;
  }

  for (int exp = 0; exp < static_cast<int>(sizeof(uint64_t)); ++exp) {
    CountingSort(neg_part, exp);
  }

  for (auto& num : neg_part) {
    DoubleWrapper dw;
    dw.d = num;
    dw.u ^= 0x8000000000000000;
    num = dw.d;
  }
  std::ranges::reverse(neg_part);

  std::vector<double> pos_part(middle.begin(), arr.end());
  for (int exp = 0; exp < static_cast<int>(sizeof(uint64_t)); ++exp) {
    CountingSort(pos_part, exp);
  }

  arr.clear();
  arr.reserve(neg_part.size() + pos_part.size());
  arr.insert(arr.end(), neg_part.begin(), neg_part.end());
  arr.insert(arr.end(), pos_part.begin(), pos_part.end());
}
}  // namespace

bool RadixSortSequential::PreProcessingImpl() {
  res_ = std::vector<double>(reinterpret_cast<double*>(task_data->inputs[0]),
                             reinterpret_cast<double*>(task_data->inputs[0]) + task_data->inputs_count[0]);
  return true;
}

bool RadixSortSequential::ValidationImpl() {
  return !task_data->inputs_count.empty() && task_data->inputs_count[0] > 0;
}

bool RadixSortSequential::RunImpl() {
  RadixSort(res_);
  return true;
}

bool RadixSortSequential::PostProcessingImpl() {
  auto* output = reinterpret_cast<double*>(task_data->outputs[0]);
  for (size_t i = 0; i < res_.size(); i++) {
    output[i] = res_[i];
  }
  return true;
}
}  // namespace malyshev_v_radix_sort_seq