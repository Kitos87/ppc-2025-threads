#include "seq/tsatsyn_a_radix_sort_simple_merge_seq/include/ops_seq.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

std::pair<std::vector<uint64_t>, std::vector<uint64_t>> tsatsyn_a_radix_sort_simple_merge_seq::ParseOrigin(
    std::vector<double> &input_data) {
  std::vector<uint64_t> pozitive_copy;
  std::vector<uint64_t> negative_copy;
  for (int i = 0; i < static_cast<int>(input_data.size()); i++) {
    if (input_data[i] > 0.0) {
      pozitive_copy.emplace_back(*reinterpret_cast<uint64_t *>(&input_data[i]));
    } else {
      negative_copy.emplace_back(*reinterpret_cast<uint64_t *>(&input_data[i]));
    }
  }
  return {pozitive_copy, negative_copy};
}
int tsatsyn_a_radix_sort_simple_merge_seq::CalculateBits(const std::vector<uint64_t> &data, bool is_pozitive) {
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
bool tsatsyn_a_radix_sort_simple_merge_seq::TestTaskSequential::PreProcessingImpl() {
  auto *temp_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  input_data_ = std::vector<double>(temp_ptr, temp_ptr + task_data->inputs_count[0]);
  output_.resize(task_data->inputs_count[0]);
  return true;
}

bool tsatsyn_a_radix_sort_simple_merge_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] != 0;
}

bool tsatsyn_a_radix_sort_simple_merge_seq::TestTaskSequential::RunImpl() {
  std::vector<uint64_t> pozitive_copy;
  std::vector<uint64_t> negative_copy;
  std::pair<std::vector<uint64_t>, std::vector<uint64_t>> temp = ParseOrigin(input_data_);
  pozitive_copy = temp.first;
  negative_copy = temp.second;
  int positive_bits = CalculateBits(pozitive_copy, true);
  int negative_bits = CalculateBits(negative_copy, false);

  for (int bit = 0; bit < positive_bits; bit++) {
    std::vector<uint64_t> group0;
    std::vector<uint64_t> group1;
    for (uint64_t b : pozitive_copy) {
      if (((b >> bit) & 1) != 0U) {
        group1.push_back(b);
      } else {
        group0.push_back(b);
      }
    }

    pozitive_copy.clear();
    pozitive_copy.insert(pozitive_copy.end(), group0.begin(), group0.end());
    pozitive_copy.insert(pozitive_copy.end(), group1.begin(), group1.end());
  }
  if (!negative_copy.empty()) {
    for (int bit = 0; bit < negative_bits; bit++) {
      std::vector<uint64_t> group0;
      std::vector<uint64_t> group1;
      for (uint64_t b : negative_copy) {
        if (((b >> bit) & 1) != 0U) {
          group1.push_back(b);
        } else {
          group0.push_back(b);
        }
      }
      negative_copy.clear();
      negative_copy.insert(negative_copy.end(), group1.begin(), group1.end());
      negative_copy.insert(negative_copy.end(), group0.begin(), group0.end());
    }
    for (int i = 0; i < static_cast<int>(negative_copy.size()); i++) {
      output_[i] = *reinterpret_cast<double *>(&negative_copy[i]);
    }
  }
  for (int i = 0; i < static_cast<int>(pozitive_copy.size()); i++) {
    output_[negative_copy.size() + i] = *reinterpret_cast<double *>(&pozitive_copy[i]);
  }
  return true;
}

bool tsatsyn_a_radix_sort_simple_merge_seq::TestTaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
