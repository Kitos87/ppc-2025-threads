#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/alputov_i_graham_scan/include/ops_seq.hpp"

namespace {
void GenerateTestData(std::vector<alputov_i_graham_scan_seq::Point>& data) {
  constexpr size_t kCount = 100000;
  constexpr double kRange = 1000.0;

  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-kRange, kRange);

  data.clear();
  data.reserve(kCount);

  data.emplace_back(-kRange, -kRange);
  data.emplace_back(kRange, -kRange);
  data.emplace_back(kRange, kRange);
  data.emplace_back(-kRange, kRange);

  for (size_t i = 4; i < kCount; ++i) {
    data.emplace_back(dist(gen), dist(gen));
  }
}
}  // namespace

TEST(alputov_i_graham_scan_seq, test_pipeline_run) {
  std::vector<alputov_i_graham_scan_seq::Point> input;
  GenerateTestData(input);
  std::vector<alputov_i_graham_scan_seq::Point> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<alputov_i_graham_scan_seq::TestTaskSequential>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  EXPECT_FALSE(task->GetConvexHull().empty());
  const auto& convex_hull = task->GetConvexHull();
  ASSERT_GT(convex_hull.size(), 3U);
  ASSERT_LE(convex_hull.size(), input.size());
}

TEST(alputov_i_graham_scan_seq, test_task_run) {
  std::vector<alputov_i_graham_scan_seq::Point> input;
  GenerateTestData(input);
  std::vector<alputov_i_graham_scan_seq::Point> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<alputov_i_graham_scan_seq::TestTaskSequential>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  EXPECT_FALSE(task->GetConvexHull().empty());
  const auto& hull = task->GetConvexHull();
  auto contains = [&hull](double x, double y) {
    return std::ranges::any_of(hull, [x, y](const auto& p) { return p.x == x && p.y == y; });
  };

  ASSERT_TRUE(contains(-1000.0, -1000.0));
  ASSERT_TRUE(contains(1000.0, -1000.0));
  ASSERT_TRUE(contains(1000.0, 1000.0));
  ASSERT_TRUE(contains(-1000.0, 1000.0));
}