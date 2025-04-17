#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/moiseev_a_mult_mat/include/ops_omp.hpp"

namespace {

std::vector<double> GenerateRandomMatrix(size_t rows, size_t cols) {
  std::vector<double> matrix(rows * cols);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-10.0, 10.0);

  for (auto &val : matrix) {
    val = dist(gen);
  }
  return matrix;
}

}  // namespace

TEST(moiseev_a_mult_mat_omp, test_pipeline_run) {
  constexpr int kCount = 500;

  auto matrix_a = GenerateRandomMatrix(kCount, kCount);
  auto matrix_b = GenerateRandomMatrix(kCount, kCount);
  std::vector<double> matrix_c(kCount * kCount, 0.0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_omp->inputs_count.emplace_back(matrix_a.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_omp->inputs_count.emplace_back(matrix_b.size());

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_c.data()));
  task_data_omp->outputs_count.emplace_back(matrix_c.size());

  auto test_task_omp = std::make_shared<moiseev_a_mult_mat_omp::MultMatOMP>(task_data_omp);

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
}

TEST(moiseev_a_mult_mat_omp, test_task_run) {
  constexpr int kCount = 500;

  auto matrix_a = GenerateRandomMatrix(kCount, kCount);
  auto matrix_b = GenerateRandomMatrix(kCount, kCount);
  std::vector<double> matrix_c(kCount * kCount, 0.0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_omp->inputs_count.emplace_back(matrix_a.size());

  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_omp->inputs_count.emplace_back(matrix_b.size());

  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_c.data()));
  task_data_omp->outputs_count.emplace_back(matrix_c.size());

  auto test_task_omp = std::make_shared<moiseev_a_mult_mat_omp::MultMatOMP>(task_data_omp);

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
}
