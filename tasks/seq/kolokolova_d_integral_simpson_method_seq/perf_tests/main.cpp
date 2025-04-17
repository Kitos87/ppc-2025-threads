#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/kolokolova_d_integral_simpson_method_seq/include/ops_seq.hpp"

TEST(kolokolova_d_integral_simpson_method_seq, test_pipeline_run) {
  auto func = [](std::vector<double> vec) {
    return (2 * vec[2]) + (vec[1] * vec[1] / 5) + (4 * vec[0] * vec[0] * vec[0]) - 100;
  };
  std::vector<int> step = {90, 90, 90};
  std::vector<int> bord = {10, 11, 8, 10, 0, 2};
  double func_result = 0.0;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
  task_data_seq->inputs_count.emplace_back(step.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
  task_data_seq->inputs_count.emplace_back(bord.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  auto test_task_sequential =
      std::make_shared<kolokolova_d_integral_simpson_method_seq::TestTaskSequential>(task_data_seq, func);

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

  double ans = 18235.7;
  double error = 0.1;
  ASSERT_NEAR(func_result, ans, error);
}

TEST(kolokolova_d_integral_simpson_method_seq, test_task_run) {
  auto func = [](std::vector<double> vec) {
    return (vec[2] * vec[2] * vec[2] * vec[1] * vec[1] / 10) + (4 * vec[0] * vec[0]) - (10 * vec[2]);
  };
  std::vector<int> step = {90, 90, 90};
  std::vector<int> bord = {10, 11, 8, 10, 0, 2};
  double func_result = 0.0;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
  task_data_seq->inputs_count.emplace_back(step.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
  task_data_seq->inputs_count.emplace_back(bord.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  auto test_task_sequential =
      std::make_shared<kolokolova_d_integral_simpson_method_seq::TestTaskSequential>(task_data_seq, func);

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
  double ans = 1790.2;
  double error = 0.1;
  ASSERT_NEAR(func_result, ans, error);
}
