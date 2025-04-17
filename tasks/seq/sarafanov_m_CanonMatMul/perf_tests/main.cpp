#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/sarafanov_m_CanonMatMul/include/ops_seq.hpp"

TEST(sarafanov_m_canon_mat_mul_seq, test_pipeline_run) {
  constexpr size_t kCount = 250;
  constexpr double kInaccuracy = 0.001;
  auto a_matrix = sarafanov_m_canon_mat_mul_seq::GenerateRandomData(static_cast<int>(kCount * kCount));
  auto single_matrix = sarafanov_m_canon_mat_mul_seq::GenerateSingleMatrix(static_cast<int>(kCount * kCount));
  std::vector<double> out(kCount * kCount, 0);
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_matrix.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(single_matrix.data()));
  for (int i = 0; i < 4; ++i) {
    task_data_seq->inputs_count.emplace_back(kCount);
  }
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(kCount * kCount);
  auto test_task_sequential = std::make_shared<sarafanov_m_canon_mat_mul_seq::CanonMatMulSequential>(task_data_seq);
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
  for (size_t i = 0; i < kCount * kCount; ++i) {
    EXPECT_NEAR(out[i], a_matrix[i], kInaccuracy);
  }
}

TEST(sarafanov_m_canon_mat_mul_seq, test_task_run) {
  constexpr size_t kCount = 250;
  constexpr double kInaccuracy = 0.001;
  auto a_matrix = sarafanov_m_canon_mat_mul_seq::GenerateRandomData(static_cast<int>(kCount * kCount));
  auto single_matrix = sarafanov_m_canon_mat_mul_seq::GenerateSingleMatrix(static_cast<int>(kCount * kCount));
  std::vector<double> out(kCount * kCount, 0);
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_matrix.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(single_matrix.data()));
  for (int i = 0; i < 4; ++i) {
    task_data_seq->inputs_count.emplace_back(kCount);
  }
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(kCount * kCount);

  auto test_task_sequential = std::make_shared<sarafanov_m_canon_mat_mul_seq::CanonMatMulSequential>(task_data_seq);

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
  for (size_t i = 0; i < kCount * kCount; ++i) {
    EXPECT_NEAR(out[i], a_matrix[i], kInaccuracy);
  }
}
