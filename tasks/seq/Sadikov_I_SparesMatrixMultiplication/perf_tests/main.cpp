#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/Sadikov_I_SparesMatrixMultiplication/include/SparesMatrix.hpp"
#include "seq/Sadikov_I_SparesMatrixMultiplication/include/ops_seq.hpp"

TEST(sadikov_i_sparse_matrix_multiplication_task_seq, test_pipeline_run) {
  constexpr auto kEpsilon = 0.00001;
  constexpr auto kSize = 300;
  auto fmatrix = sadikov_i_sparse_matrix_multiplication_task_seq::GetRandomMatrix(kSize * kSize);
  auto smatrix = sadikov_i_sparse_matrix_multiplication_task_seq::GetRandomMatrix(kSize * kSize);
  std::vector<double> out(kSize * kSize, 0.0);
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(fmatrix.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(smatrix.data()));
  task_data_seq->inputs_count.emplace_back(kSize);
  task_data_seq->inputs_count.emplace_back(kSize);
  task_data_seq->inputs_count.emplace_back(kSize);
  task_data_seq->inputs_count.emplace_back(kSize);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  auto check_out = sadikov_i_sparse_matrix_multiplication_task_seq::BaseMatrixMultiplication(fmatrix, kSize, kSize,
                                                                                             smatrix, kSize, kSize);
  auto test_task_sequential =
      std::make_shared<sadikov_i_sparse_matrix_multiplication_task_seq::CCSMatrixSequential>(task_data_seq);
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
  for (auto i = 0; i < static_cast<int>(out.size()); ++i) {
    EXPECT_NEAR(out[i], check_out[i], kEpsilon);
  }
}

TEST(sadikov_i_sparse_matrix_multiplication_task_seq, test_task_run) {
  constexpr auto kEpsilon = 0.00001;
  constexpr auto kSize = 300;
  auto fmatrix = sadikov_i_sparse_matrix_multiplication_task_seq::GetRandomMatrix(kSize * kSize);
  auto smatrix = sadikov_i_sparse_matrix_multiplication_task_seq::GetRandomMatrix(kSize * kSize);
  std::vector<double> out(kSize * kSize, 0.0);
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(fmatrix.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(smatrix.data()));
  task_data_seq->inputs_count.emplace_back(kSize);
  task_data_seq->inputs_count.emplace_back(kSize);
  task_data_seq->inputs_count.emplace_back(kSize);
  task_data_seq->inputs_count.emplace_back(kSize);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());
  auto check_out = sadikov_i_sparse_matrix_multiplication_task_seq::BaseMatrixMultiplication(fmatrix, kSize, kSize,
                                                                                             smatrix, kSize, kSize);
  auto test_task_sequential =
      std::make_shared<sadikov_i_sparse_matrix_multiplication_task_seq::CCSMatrixSequential>(task_data_seq);
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
  for (auto i = 0; i < static_cast<int>(out.size()); ++i) {
    EXPECT_NEAR(out[i], check_out[i], kEpsilon);
  }
}
