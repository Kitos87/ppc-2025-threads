#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "../include/ops.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {
inline constexpr std::size_t kDataSize = 24000000;

template <typename MiddlewareFn>
void PerfTest(MiddlewareFn middleware) {
  std::vector<double> in(kDataSize);
  std::iota(in.rbegin(), in.rend(), 0);
  decltype(in) out(in.size());

  auto data = std::make_shared<ppc::core::TaskData>(ppc::core::TaskData{
      .inputs = {reinterpret_cast<uint8_t *>(in.data())},
      .inputs_count = {static_cast<std::uint32_t>(in.size())},
      .outputs = {reinterpret_cast<uint8_t *>(out.data())},
      .outputs_count = {static_cast<unsigned int>(out.size())},
      .state_of_testing = {},
  });

  auto task = std::make_shared<sorochkin_d_radix_double_sort_simple_merge_seq::SortTask>(data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  ppc::core::Perf perf_analyzer(task);
  middleware(perf_analyzer, perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  EXPECT_TRUE(std::ranges::is_sorted(out));
}
}  // namespace

TEST(sorochkin_d_radix_double_sort_simple_merge_seq, test_pipeline_run) {
  PerfTest([](const auto &perf_analyzer, const auto &perf_attr, const auto &perf_results) {
    perf_analyzer.PipelineRun(perf_attr, perf_results);
  });
}

TEST(sorochkin_d_radix_double_sort_simple_merge_seq, test_task_run) {
  PerfTest([](const auto &perf_analyzer, const auto &perf_attr, const auto &perf_results) {
    perf_analyzer.TaskRun(perf_attr, perf_results);
  });
}
