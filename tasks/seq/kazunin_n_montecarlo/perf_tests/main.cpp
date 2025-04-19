#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/kazunin_n_montecarlo/include/ops_seq.hpp"

using std::cos;
using std::sin;

TEST(kazunin_n_montecarlo_seq, test_pipeline_run) {
  double out = 0.0;

  const std::size_t n = 10;
  std::array<std::pair<double, double>, n> limits;
  std::ranges::fill(limits, std::make_pair(0.0, 1.0));
  std::size_t precision = 4200000;
  const auto f = [](const std::array<double, n> &args) {
    return std::accumulate(args.begin(), args.end(), 1.0,
                           [](const double acc, const double coord) { return acc + (sin(coord) * cos(coord)); });
  };  // ((sin(x)*cos(x)) + (sin(y)*cos(y)) + ...)

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t *>(&precision), reinterpret_cast<uint8_t *>(&limits)};
  task_data->inputs_count = {1, n};
  task_data->outputs = {reinterpret_cast<uint8_t *>(&out)};
  task_data->outputs_count = {1};

  // Create Task
  auto test_task = std::make_shared<kazunin_n_montecarlo_seq::MonteCarloSeq<n, decltype(f)>>(task_data, f);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(kazunin_n_montecarlo_seq, test_task_run) {
  double out = 0.0;

  const std::size_t n = 10;
  std::array<std::pair<double, double>, n> limits;
  std::ranges::fill(limits, std::make_pair(0.0, 1.0));
  std::size_t precision = 4200000;
  const auto f = [](const std::array<double, n> &args) {
    return std::accumulate(args.begin(), args.end(), 1.0,
                           [](const double acc, const double coord) { return acc + (sin(coord) * cos(coord)); });
  };  // ((sin(x)*cos(x)) + (sin(y)*cos(y)) + ...)

  // Create task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t *>(&precision), reinterpret_cast<uint8_t *>(&limits)};
  task_data->inputs_count = {1, n};
  task_data->outputs = {reinterpret_cast<uint8_t *>(&out)};
  task_data->outputs_count = {1};

  // Create Task
  auto test_task = std::make_shared<kazunin_n_montecarlo_seq::MonteCarloSeq<n, decltype(f)>>(task_data, f);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}
