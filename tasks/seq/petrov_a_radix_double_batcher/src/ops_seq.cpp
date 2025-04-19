#include "seq/petrov_a_radix_double_batcher/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <vector>

namespace {
auto Translate(double e, size_t i) {
  const uint64_t mask = 1ULL << ((sizeof(uint64_t) * 8) - 1);
  const union {
    double dbl;
    uint64_t uint64;
  } uni{.dbl = e};
  const uint64_t u = ((uni.uint64 & mask) == 0U) ? (uni.uint64 | mask) : (~uni.uint64);
  return (u >> (i * 8)) & 0xFF;
}
}  // namespace

bool petrov_a_radix_double_batcher_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool petrov_a_radix_double_batcher_seq::TestTaskSequential::PreProcessingImpl() {
  in_.assign(reinterpret_cast<double *>(task_data->inputs[0]),
             reinterpret_cast<double *>(task_data->inputs[0]) + task_data->inputs_count[0]);
  return true;
}

bool petrov_a_radix_double_batcher_seq::TestTaskSequential::RunImpl() {
  res_.resize(in_.size());
  std::ranges::copy(in_, res_.begin());

  std::vector<double> tmp(res_.size());
  std::vector<size_t> cnt(1 << 8, 0);

  for (size_t i = 0; i < sizeof(double); ++i) {
    for (auto &c : cnt) {
      c = 0;
    }
    for (auto &e : res_) {
      ++cnt[Translate(e, i)];
    }
    for (size_t j = 1; j < cnt.size(); ++j) {
      cnt[j] += cnt[j - 1];
    }
    for (auto &e : res_ | std::views::reverse) {
      const auto nidx = --cnt[Translate(e, i)];
      tmp[nidx] = e;
    }
    std::swap(res_, tmp);
  }

  return true;
}

bool petrov_a_radix_double_batcher_seq::TestTaskSequential::PostProcessingImpl() {
  std::ranges::copy(res_, reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}
