#include "../include/ops_seq.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
void RadixIntegerSortHomogenous(std::vector<int> &arr) {
  using TInt = std::decay_t<decltype(arr)>::value_type;

  const auto convert = [](TInt num, std::size_t i) {
    constexpr TInt kMask = 1 << ((sizeof(TInt) * CHAR_BIT) - 1);
    num = ((num & kMask) == 0U) ? (num | kMask) : (~num);
    return (num >> (i * 8)) & 0xFF;
  };

  std::vector<TInt> buf(arr.size());
  std::size_t counts[1 << 8];

  for (std::size_t pp = 0; pp < sizeof(TInt); ++pp) {
    std::ranges::fill(counts, 0);
    for (auto &num : arr) {
      ++counts[convert(num, pp)];
    }
    std::partial_sum(std::begin(counts), std::end(counts), std::begin(counts));
    for (auto &num : std::views::reverse(arr)) {
      buf[--counts[convert(num, pp)]] = num;
    }
    std::swap(arr, buf);
  }
}

std::vector<int> RadixIntegerSort(const std::vector<int> &arr) {
  std::vector<int> neg;
  std::vector<int> pos;

  for (auto num : arr) {
    if (num < 0) {
      neg.push_back(num);
    } else {
      pos.push_back(num);
    }
  }
  RadixIntegerSortHomogenous(neg);
  RadixIntegerSortHomogenous(pos);

  std::vector<int> res(arr.size());
  std::ranges::reverse_copy(neg, res.begin());
  std::ranges::copy(pos, res.begin() + static_cast<decltype(res)::difference_type>(neg.size()));
  return res;
}
}  // namespace

bool koshkin_m_radix_int_simple_merge::SeqT::PreProcessingImpl() {
  const auto &[src, cnt] =
      std::pair(reinterpret_cast<decltype(in_)::value_type *>(task_data->inputs[0]), task_data->inputs_count[0]);
  in_.assign(src, src + cnt);
  return true;
}

bool koshkin_m_radix_int_simple_merge::SeqT::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool koshkin_m_radix_int_simple_merge::SeqT::RunImpl() {
  out_ = RadixIntegerSort(in_);
  return true;
}

bool koshkin_m_radix_int_simple_merge::SeqT::PostProcessingImpl() {
  auto *tgt = reinterpret_cast<decltype(out_)::value_type *>(task_data->outputs[0]);
  std::ranges::copy(out_, tgt);
  return true;
}
