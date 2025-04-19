#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/shuravina_o_hoare_simple_merger_omp1/include/ops_omp.hpp"

TEST(shuravina_o_hoare_simple_merger_omp, test_pipeline_run) {
  constexpr int kCount = 50000;

  std::vector<int> in(kCount, 0);
  std::vector<int> out(kCount, 0);

  for (int i = 0; i < kCount; i++) {
    in[i] = kCount - i;
  }

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  auto test_task_omp = std::make_shared<shuravina_o_hoare_simple_merger::TestTaskOMP>(task_data_omp);

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

  std::vector<int> expected(kCount);
  for (int i = 0; i < kCount; i++) {
    expected[i] = i + 1;
  }
  EXPECT_EQ(out, expected);
}

TEST(shuravina_o_hoare_simple_merger_omp, test_task_run) {
  constexpr int kCount = 50000;

  std::vector<int> in(kCount, 0);
  std::vector<int> out(kCount, 0);

  for (int i = 0; i < kCount; i++) {
    in[i] = kCount - i;
  }

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  auto test_task_omp = std::make_shared<shuravina_o_hoare_simple_merger::TestTaskOMP>(task_data_omp);

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

  std::vector<int> expected(kCount);
  for (int i = 0; i < kCount; i++) {
    expected[i] = i + 1;
  }
  EXPECT_EQ(out, expected);
}