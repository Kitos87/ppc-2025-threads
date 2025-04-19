
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/odintsov_m_multmatrix_cannon/include/ops_seq.hpp"

TEST(odintsov_m_multmatrix_cannon_seq, test_pipeline_run) {
  // Create data
  std::vector<double> matrix_a(90000, 1);
  std::vector<double> matrix_b(90000, 1);
  std::vector<double> out(90000, 0);
  std::vector<double> matrix_c(90000, 300);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_seq->inputs_count.emplace_back(matrix_a.size());
  task_data_seq->inputs_count.emplace_back(matrix_a.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));

  // Create Task
  auto test_task_sequential =
      std::make_shared<odintsov_m_mulmatrix_cannon_seq::MulMatrixCannonSequential>(task_data_seq);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 100;
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
  EXPECT_EQ(out, matrix_c);
}

TEST(odintsov_m_multmatrix_cannon_seq, test_task_run) {
  // Create data
  std::vector<double> matrix_a(90000, 1);
  std::vector<double> matrix_b(90000, 1);
  std::vector<double> out(90000, 0);
  std::vector<double> matrix_c(90000, 300);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_seq->inputs_count.emplace_back(matrix_a.size());
  task_data_seq->inputs_count.emplace_back(matrix_a.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));

  // Create Task
  auto test_task_sequential =
      std::make_shared<odintsov_m_mulmatrix_cannon_seq::MulMatrixCannonSequential>(task_data_seq);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 100;
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
  EXPECT_EQ(out, matrix_c);
}
