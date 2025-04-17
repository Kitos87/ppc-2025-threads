#include "omp/tsatsyn_a_radix_sort_simple_merge_omp/include/ops_omp.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace constants {
constexpr int kChunk = 100;
}  // namespace constants
inline std::vector<uint64_t> tsatsyn_a_radix_sort_simple_merge_omp::MainSort(std::vector<uint64_t> &data, int bit) {
  std::vector<uint64_t> group0;
  std::vector<uint64_t> group1;

  group0.reserve(data.size());
  group1.reserve(data.size());
#pragma omp for schedule(guided, constants::kChunk) nowait
  for (int i = 0; i < static_cast<int>(data.size()); i++) {
    (((data[i] >> bit) & 1) != 0U) ? group1.push_back(data[i]) : group0.push_back(data[i]);
  }
  data = std::move(group0);
  data.insert(data.end(), group1.begin(), group1.end());

  return data;
}

inline int tsatsyn_a_radix_sort_simple_merge_omp::CalculateBits(const std::vector<uint64_t> &data, bool is_pozitive) {
  if (data.empty()) {
    return 0;
  }
  uint64_t extreme_val = 0;
  int num_bits = 0;
  if (is_pozitive) {
    extreme_val = *std::ranges::max_element(data);
    num_bits = std::bit_width(extreme_val);
  } else {
    extreme_val = *std::ranges::min_element(data);
    num_bits = (extreme_val == 0) ? 0 : std::bit_width(extreme_val);
  }

  return num_bits;
}
bool tsatsyn_a_radix_sort_simple_merge_omp::TestTaskOpenMP::PreProcessingImpl() {
  auto *temp_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  input_data_ = std::vector<double>(temp_ptr, temp_ptr + task_data->inputs_count[0]);
  output_.resize(task_data->inputs_count[0]);
  return true;
}

bool tsatsyn_a_radix_sort_simple_merge_omp::TestTaskOpenMP::ValidationImpl() { return task_data->inputs_count[0] != 0; }

bool tsatsyn_a_radix_sort_simple_merge_omp::TestTaskOpenMP::RunImpl() {
  std::vector<uint64_t> pozitive_copy;
  std::vector<uint64_t> negative_copy;
  int i = 0;
#pragma omp parallel private(i)
  {
    std::vector<uint64_t> local_positive;
    std::vector<uint64_t> local_negative;

#pragma omp for nowait
    for (i = 0; i < static_cast<int>(input_data_.size()); ++i) {
      if (input_data_[i] > 0.0) {
        local_positive.push_back(*reinterpret_cast<const uint64_t *>(&input_data_[i]));
      } else {
        local_negative.push_back(*reinterpret_cast<const uint64_t *>(&input_data_[i]));
      }
    }
#pragma omp critical
    {
      pozitive_copy.insert(pozitive_copy.end(), local_positive.begin(), local_positive.end());
      negative_copy.insert(negative_copy.end(), local_negative.begin(), local_negative.end());
    }
  }
  int pozitive_bits = CalculateBits(pozitive_copy, true);
  int negative_bits = CalculateBits(negative_copy, false);

  for (int bit = 0; bit < pozitive_bits; bit++) {
    pozitive_copy = MainSort(pozitive_copy, bit);
  }

  if (!negative_copy.empty()) {
    for (int bit = 0; bit < negative_bits; bit++) {
      negative_copy = MainSort(negative_copy, bit);
    }

#pragma omp parallel for schedule(guided, constants::kChunk)
    for (i = 0; i < static_cast<int>(negative_copy.size()); i++) {
      output_[static_cast<int>(negative_copy.size()) - 1 - i] = *reinterpret_cast<const double *>(&negative_copy[i]);
    }
  }
#pragma omp parallel for schedule(guided, constants::kChunk)
  for (i = 0; i < static_cast<int>(pozitive_copy.size()); ++i) {
    output_[negative_copy.size() + i] = *reinterpret_cast<const double *>(&pozitive_copy[i]);
  }

  return true;
}
bool tsatsyn_a_radix_sort_simple_merge_omp::TestTaskOpenMP::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
