#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/tyshkevich_a_hoare_simple_merge/include/ops_seq.hpp"

constexpr size_t kTestSequenceSize = 55555;

namespace {
template <typename T>
std::vector<T> GenUnsortedSequence(size_t size) {
  std::vector<T> vec(size);
  std::iota(vec.rbegin(), vec.rend(), 0);
  return vec;
}
}  // namespace

TEST(tyshkevich_a_hoare_simple_merge_seq, test_pipeline_run) {
  auto in = GenUnsortedSequence<int>(kTestSequenceSize);
  std::vector<int> out(in.size());

  auto dat = std::make_shared<ppc::core::TaskData>();
  dat->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  dat->inputs_count.emplace_back(in.size());
  dat->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  dat->outputs_count.emplace_back(out.size());

  auto stt = tyshkevich_a_hoare_simple_merge_seq::CreateHoareTestTask<int>(dat, std::greater<>());

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(std::make_shared<decltype(stt)>(std::move(stt)));
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_TRUE(std::ranges::is_sorted(out, std::greater<>()));
}

TEST(tyshkevich_a_hoare_simple_merge_seq, test_task_run) {
  auto in = GenUnsortedSequence<int>(kTestSequenceSize);
  std::vector<int> out(in.size());

  auto dat = std::make_shared<ppc::core::TaskData>();
  dat->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  dat->inputs_count.emplace_back(in.size());
  dat->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  dat->outputs_count.emplace_back(out.size());

  auto stt = tyshkevich_a_hoare_simple_merge_seq::CreateHoareTestTask<int>(dat, std::greater<>());

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(std::make_shared<decltype(stt)>(std::move(stt)));
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_TRUE(std::ranges::is_sorted(out, std::greater<>()));
}
