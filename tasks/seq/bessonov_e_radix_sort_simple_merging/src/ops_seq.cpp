#include "seq/bessonov_e_radix_sort_simple_merging/include/ops_seq.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

bool bessonov_e_radix_sort_simple_merging_seq::TestTaskSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  input_.assign(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_.resize(output_size);

  return true;
}

bool bessonov_e_radix_sort_simple_merging_seq::TestTaskSequential::ValidationImpl() {
  if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
    return false;
  }

  if (task_data->inputs_count[0] == 0) {
    return false;
  }

  if (task_data->inputs_count[0] != task_data->outputs_count[0]) {
    return false;
  }

  return true;
}

bool bessonov_e_radix_sort_simple_merging_seq::TestTaskSequential::RunImpl() {
  size_t n = input_.size();
  std::vector<uint64_t> bits(n);

  for (size_t i = 0; i < n; i++) {
    uint64_t b = 0;
    std::memcpy(&b, &input_[i], sizeof(double));
    if ((b & (1ULL << 63)) != 0ULL) {
      b = ~b;
    } else {
      b ^= (1ULL << 63);
    }
    bits[i] = b;
  }

  const int radix = 256;
  const int passes = 8;
  std::vector<uint64_t> temp(n);

  for (int pass = 0; pass < passes; pass++) {
    int shift = pass * 8;
    std::vector<size_t> count(radix, 0);

    for (size_t i = 0; i < n; i++) {
      int digit = static_cast<int>((bits[i] >> shift) & 0xFF);
      count[digit]++;
    }
    for (int i = 1; i < radix; i++) {
      count[i] += count[i - 1];
    }
    for (size_t i = n; i-- > 0;) {
      int digit = static_cast<int>((bits[i] >> shift) & 0xFF);
      temp[--count[digit]] = bits[i];
    }
    bits.swap(temp);
  }

  for (size_t i = 0; i < n; i++) {
    uint64_t b = bits[i];
    if ((b & (1ULL << 63)) != 0ULL) {
      b ^= (1ULL << 63);
    } else {
      b = ~b;
    }
    double d = 0.0;
    std::memcpy(&d, &b, sizeof(double));
    output_[i] = d;
  }

  return true;
}

bool bessonov_e_radix_sort_simple_merging_seq::TestTaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}