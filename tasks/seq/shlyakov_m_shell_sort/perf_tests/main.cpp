#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/shlyakov_m_shell_sort/include/ops_seq.hpp"

namespace {
std::vector<int> GenerateRandomArray(size_t size) {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);

  std::uniform_int_distribution<int> distribution_range(-1000, 1000);
  int min_val = distribution_range(generator);
  int max_val = distribution_range(generator);

  if (min_val > max_val) {
    std::swap(min_val, max_val);
  }

  std::uniform_int_distribution<int> distribution(min_val, max_val);

  std::vector<int> arr(size);
  for (size_t i = 0; i < size; ++i) {
    arr[i] = distribution(generator);
  }
  return arr;
}

bool IsSorted(const std::vector<int>& arr) {
  if (arr.empty()) {
    return true;
  }
  for (size_t i = 1; i < arr.size(); ++i) {
    if (arr[i - 1] > arr[i]) {
      return false;
    }
  }
  return true;
}
}  // namespace

TEST(shlyakov_m_shell_sort_seq, test_pipeline_run) {
  constexpr size_t kCount = 50000;

  std::vector<int> in = GenerateRandomArray(kCount);
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  std::vector<int> out(in.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<shlyakov_m_shell_sort_seq::TestTaskSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);

  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  EXPECT_TRUE(IsSorted(out));
}

TEST(shlyakov_m_shell_sort_seq, test_task_run) {
  constexpr size_t kCount = 50000;

  std::vector<int> in = GenerateRandomArray(kCount);
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  std::vector<int> out(in.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<shlyakov_m_shell_sort_seq::TestTaskSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);

  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  EXPECT_TRUE(IsSorted(out));
}