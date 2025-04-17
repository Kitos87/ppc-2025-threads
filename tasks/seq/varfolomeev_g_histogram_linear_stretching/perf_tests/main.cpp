#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/varfolomeev_g_histogram_linear_stretching/include/ops_seq.hpp"

namespace {
std::vector<uint8_t> GetRandomImage(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> dis(0, 255);
  std::vector<uint8_t> res(sz);

  for (int i = 0; i < sz; ++i) {
    res[i] = dis(gen);
  }

  return res;
}
}  // namespace

TEST(varfolomeev_g_histogram_linear_stretching_seq, test_pipeline_run) {
  constexpr int kCount = 10000000;

  // Create data
  std::vector<uint8_t> in = GetRandomImage(kCount);
  std::vector<uint8_t> out(kCount);
  std::vector<uint8_t> expected_out(kCount);

  int min_val = *std::ranges::min_element(in);
  int max_val = *std::ranges::max_element(in);
  if (min_val != max_val) {
    for (size_t i = 0; i < in.size(); ++i) {
      expected_out[i] = static_cast<int>(std::round((in[i] - min_val) / static_cast<double>(max_val - min_val) * 255));
    }
  }

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_sequential =
      std::make_shared<varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential>(task_data_seq);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(expected_out, out);
}

TEST(varfolomeev_g_histogram_linear_stretching_seq, test_task_run) {
  constexpr int kCount = 10000000;

  // Create data
  std::vector<uint8_t> in = GetRandomImage(kCount);
  std::vector<uint8_t> out(kCount, 0);
  std::vector<uint8_t> expected_out(kCount, 0);

  int min_val = *std::ranges::min_element(in);
  int max_val = *std::ranges::max_element(in);
  if (min_val != max_val) {
    for (size_t i = 0; i < in.size(); ++i) {
      expected_out[i] = static_cast<int>(std::round((in[i] - min_val) / static_cast<double>(max_val - min_val) * 255));
    }
  }

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_sequential =
      std::make_shared<varfolomeev_g_histogram_linear_stretching_seq::TestTaskSequential>(task_data_seq);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(expected_out, out);
}
