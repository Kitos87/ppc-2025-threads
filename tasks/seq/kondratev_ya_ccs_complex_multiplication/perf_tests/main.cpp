#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/kondratev_ya_ccs_complex_multiplication/include/ops_seq.hpp"

TEST(kondratev_ya_ccs_complex_multiplication_seq, test_pipeline_run) {
  constexpr int kCount = 27000;

  kondratev_ya_ccs_complex_multiplication_seq::CCSMatrix a({kCount, kCount});
  kondratev_ya_ccs_complex_multiplication_seq::CCSMatrix b({kCount, kCount});
  kondratev_ya_ccs_complex_multiplication_seq::CCSMatrix c({kCount, kCount});

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  task_data_seq->inputs_count.emplace_back(2);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&c));
  task_data_seq->outputs_count.emplace_back(1);

  auto test_task_sequential =
      std::make_shared<kondratev_ya_ccs_complex_multiplication_seq::TestTaskSequential>(task_data_seq);

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
}

TEST(kondratev_ya_ccs_complex_multiplication_seq, test_task_run) {
  constexpr int kCount = 27000;

  kondratev_ya_ccs_complex_multiplication_seq::CCSMatrix a({kCount, kCount});
  kondratev_ya_ccs_complex_multiplication_seq::CCSMatrix b({kCount, kCount});
  kondratev_ya_ccs_complex_multiplication_seq::CCSMatrix c({kCount, kCount});

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  task_data_seq->inputs_count.emplace_back(2);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&c));
  task_data_seq->outputs_count.emplace_back(1);

  auto test_task_sequential =
      std::make_shared<kondratev_ya_ccs_complex_multiplication_seq::TestTaskSequential>(task_data_seq);

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
}
