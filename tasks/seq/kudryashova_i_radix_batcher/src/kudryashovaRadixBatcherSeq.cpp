#include "seq/kudryashova_i_radix_batcher/include/kudryashovaRadixBatcherSeq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

void kudryashova_i_radix_batcher_seq::RadixDoubleSort(std::vector<double> &data, int first, int last) {
  const int sort_size = last - first;
  std::vector<uint64_t> converted(sort_size);
  // Convert each double to uint64_t representation
  for (int i = 0; i < sort_size; ++i) {
    double value = data[first + i];
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(value));
    converted[i] = ((bits & (1ULL << 63)) != 0) ? ~bits : bits ^ (1ULL << 63);  // sign of the number
  }
  std::vector<uint64_t> buffer(sort_size);
  int bits_int_byte = 8;
  int total_passes = sizeof(uint64_t);  // Total number of passes based on uint64_t size
  int max_byte_value = 255;

  for (int shift = 0; shift < total_passes; ++shift) {
    size_t count[256] = {0};                      // Array to count occurrences of each byte
    const int shift_loc = shift * bits_int_byte;  // Determine shift for the current pass

    // Count occurrences of each byte in the current shift position
    for (const auto &num : converted) {
      ++count[(num >> shift_loc) & max_byte_value];
    }

    size_t total = 0;
    // Convert the count array to a prefix sum array
    for (auto &safe : count) {
      size_t old = safe;
      safe = total;
      total += old;
    }
    // Rearrange the elements based on the prefix sums
    for (const auto &num : converted) {
      const uint8_t byte = (num >> shift_loc) & max_byte_value;
      buffer[count[byte]++] = num;
    }
    converted.swap(buffer);
  }
  // Convert the sorted uint64_t representations back to double
  for (int i = 0; i < sort_size; ++i) {
    uint64_t bits = converted[i];
    bits = ((bits & (1ULL << 63)) != 0) ? (bits ^ (1ULL << 63)) : ~bits;
    std::memcpy(&data[first + i], &bits, sizeof(double));
  }
}

bool kudryashova_i_radix_batcher_seq::TestTaskSequential::PreProcessingImpl() {
  input_data_.resize(task_data->inputs_count[0]);
  if (task_data->inputs[0] == nullptr || task_data->inputs_count[0] == 0) {
    return false;
  }
  auto *tmp_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + task_data->inputs_count[0], input_data_.begin());
  return true;
}

bool kudryashova_i_radix_batcher_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] > 0 && task_data->outputs_count[0] == task_data->inputs_count[0];
}

bool kudryashova_i_radix_batcher_seq::TestTaskSequential::RunImpl() {
  RadixDoubleSort(input_data_, 0, static_cast<int>(input_data_.size()));
  return true;
}

bool kudryashova_i_radix_batcher_seq::TestTaskSequential::PostProcessingImpl() {
  std::ranges::copy(input_data_, reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}
