#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/karaseva_e_congrad/include/ops_seq.hpp"

TEST(karaseva_e_congrad_seq, test_pipeline_run) {
  constexpr int kSize = 10000;

  // Create matrix a (identity matrix) and vector b (all 1)
  std::vector<double> a(kSize * kSize, 0.0);
  std::vector<double> b(kSize, 1.0);
  std::vector<double> x(kSize, 0.0);  // output vector for the solution

  for (int i = 0; i < kSize; ++i) {
    a[(i * kSize) + i] = 1.0;
  }

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_seq->inputs_count.emplace_back(a.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.emplace_back(b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(x.data()));
  task_data_seq->outputs_count.emplace_back(x.size());

  // Create Task
  auto test_task_sequential = std::make_shared<karaseva_e_congrad_seq::TestTaskSequential>(task_data_seq);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Initialize performance results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Performance analyzer using PipelineRun
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_EQ(b, x);
}

TEST(karaseva_e_congrad_seq, test_task_run) {
  constexpr int kSize = 12000;

  // Create matrix a (identity matrix) and vector b (all ones)
  std::vector<double> a(kSize * kSize, 0.0);
  std::vector<double> b(kSize, 1.0);
  std::vector<double> x(kSize, 0.0);  // output vector for the solution

  for (int i = 0; i < kSize; ++i) {
    a[(i * kSize) + i] = 1.0;
  }

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(a.data()));
  task_data_seq->inputs_count.emplace_back(a.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.emplace_back(b.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(x.data()));
  task_data_seq->outputs_count.emplace_back(x.size());

  // Create Task
  auto test_task_sequential = std::make_shared<karaseva_e_congrad_seq::TestTaskSequential>(task_data_seq);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Initialize performance results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Performance analyzer using TaskRun
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_EQ(b, x);
}