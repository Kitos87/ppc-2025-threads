#include <gtest/gtest.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/sharamygina_i_multi_dim_monte_carlo/include/ops_seq.h"

namespace {
std::vector<double> GetBoundaries(double left, double right, unsigned int dimension) {
  std::vector<double> v(dimension * 2);
  for (unsigned int i = 0; i < dimension; i++) {
    v[i * 2] = left;
    v[(i * 2) + 1] = right;
  }
  return v;
}
}  // namespace

TEST(sharamygina_i_multi_dim_monte_carlo_seq, test_pipeline_run) {
  int iterations = 15000000;
  std::vector<double> boundaries = GetBoundaries(0.0, 1.2, 7);
  auto test_function = [](const std::vector<double>& values) {
    assert(values.size() == 3);
    return (values[0] * values[3]) + values[2] - (0.3 * values[1]) + std::sin(values[4]) +
           std::exp(values[5] + values[6]);
  };
  std::function<double(const std::vector<double>&)> function_ptr = test_function;

  auto task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(boundaries.data()));
  task_data->inputs_count.emplace_back(boundaries.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&function_ptr));
  task_data->inputs_count.emplace_back(1);

  double result = 0.0;
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(1);

  auto task = std::make_shared<sharamygina_i_multi_dim_monte_carlo_seq::MultiDimMonteCarloTask>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 1;

  const auto start = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&]() {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(task);

  perf->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  double expected = 18.093354135967223;
  double tol = 0.03 * expected;
  EXPECT_NEAR(result, expected, tol);
}

TEST(sharamygina_i_multi_dim_monte_carlo_seq, test_task_run) {
  int iterations = 15000000;
  std::vector<double> boundaries = GetBoundaries(0.0, 1.2, 7);
  auto test_function = [](const std::vector<double>& values) {
    assert(values.size() == 3);
    return (values[0] * values[3]) + values[2] - (0.3 * values[1]) + std::sin(values[4]) +
           std::exp(values[5] + values[6]);
  };
  std::function<double(const std::vector<double>&)> function_ptr = test_function;

  auto task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(boundaries.data()));
  task_data->inputs_count.emplace_back(boundaries.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&function_ptr));
  task_data->inputs_count.emplace_back(1);

  double result = 0.0;
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(1);

  auto task = std::make_shared<sharamygina_i_multi_dim_monte_carlo_seq::MultiDimMonteCarloTask>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 1;

  const auto start = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&]() {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(task);

  perf->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  double expected = 18.093354135967223;
  double tol = 0.03 * expected;
  EXPECT_NEAR(result, expected, tol);
}
