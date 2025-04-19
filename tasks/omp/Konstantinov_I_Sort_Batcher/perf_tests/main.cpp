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
#include "omp/Konstantinov_I_Sort_Batcher/include/ops_omp.hpp"

TEST(Konstantinov_I_Sort_Batcher_omp, test_pipeline_run) {
  constexpr int kCount = 300000;

  std::vector<double> in(kCount);
  std::vector<double> exp_out(kCount);
  std::vector<double> out(kCount);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-10000.0, 10000.0);

  for (size_t i = 0; i < kCount; i++) {
    in[i] = dist(gen);
    exp_out[i] = in[i];
  }

  std::ranges::sort(exp_out);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  auto test_task_omp = std::make_shared<konstantinov_i_sort_batcher_omp::RadixSortBatcherOmp>(task_data_omp);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_omp);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_EQ(exp_out, out);
}

TEST(Konstantinov_I_Sort_Batcher_omp, test_task_run) {
  constexpr int kCount = 300000;

  std::vector<double> in(kCount);
  std::vector<double> exp_out(kCount);
  std::vector<double> out(kCount);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-10000.0, 10000.0);

  for (size_t i = 0; i < kCount; i++) {
    in[i] = dist(gen);
    exp_out[i] = in[i];
  }
  std::ranges::sort(exp_out);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  auto test_task_omp = std::make_shared<konstantinov_i_sort_batcher_omp::RadixSortBatcherOmp>(task_data_omp);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_omp);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_EQ(exp_out, out);
}