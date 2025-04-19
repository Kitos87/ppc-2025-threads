#include "../include/ops.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <ranges>
#include <span>
#include <vector>

namespace {
template <typename T>
constexpr size_t Bytes() {
  return sizeof(T);
}
template <typename T>
constexpr size_t Bits() {
  return Bytes<T>() * CHAR_BIT;
}
class Bitutil {
 private:
  union du64 {
    double d;
    uint64_t u;
    static constexpr uint64_t kMask = 1ULL << ((sizeof(uint64_t) * CHAR_BIT) - 1);
  };

 public:
  static constexpr uint64_t AsU64(double x) {
    const du64 r{.d = x};
    return ((r.u & du64::kMask) != 0U) ? ~r.u : r.u | du64::kMask;
  }

  template <typename T>
    requires std::is_floating_point_v<T> or std::is_integral_v<T>
  static constexpr uint8_t ByteAt(const T &val, uint8_t idx) {
    return (val >> (idx * 8)) & 0xFF;
  }
};

void RadixSort(std::span<double> v) {
  constexpr size_t kBase = 1 << CHAR_BIT;

  std::vector<double> aux_buf(v.size());
  std::span<double> aux{aux_buf};

  std::array<std::size_t, kBase> count;

  for (std::size_t ib = 0; ib < Bytes<double>(); ++ib) {
    std::ranges::fill(count, 0);
    std::ranges::for_each(v, [&](auto el) { ++count[Bitutil::ByteAt(Bitutil::AsU64(el), ib)]; });
    std::partial_sum(count.begin(), count.end(), count.begin());
    std::ranges::for_each(std::ranges::reverse_view(v),
                          [&](auto el) { aux[--count[Bitutil::ByteAt(Bitutil::AsU64(el), ib)]] = el; });
    std::swap(v, aux);
  }
}
}  // namespace

bool sorochkin_d_radix_double_sort_simple_merge_seq::SortTask::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool sorochkin_d_radix_double_sort_simple_merge_seq::SortTask::PreProcessingImpl() {
  std::span<double> src = {reinterpret_cast<double *>(task_data->inputs[0]), task_data->inputs_count[0]};
  input_.assign(src.begin(), src.end());
  return true;
}

bool sorochkin_d_radix_double_sort_simple_merge_seq::SortTask::RunImpl() {
  output_ = input_;
  RadixSort(output_);
  return true;
}

bool sorochkin_d_radix_double_sort_simple_merge_seq::SortTask::PostProcessingImpl() {
  std::ranges::copy(output_, reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}
