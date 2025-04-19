#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/sarafanov_m_CanonMatMul_omp/include/ops_omp.hpp"

namespace {
std::vector<double> GenerateRandomData(int size) {
  std::vector<double> matrix(size);
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> dist(-500, 10000);
  for (auto i = 0; i < size; ++i) {
    matrix[i] = static_cast<double>(dist(gen));
  }
  return matrix;
}

std::vector<double> GenerateSingleMatrix(int size) {
  std::vector<double> matrix(size, 0.0);
  int sqrt_size = static_cast<int>(std::sqrt(size));
  for (int i = 0; i < sqrt_size; ++i) {
    for (int j = 0; j < sqrt_size; ++j) {
      if (i == j) {
        matrix[(sqrt_size * i) + j] = 1.0;
      }
    }
  }
  return matrix;
}
}  // namespace

TEST(sarafanov_m_canon_mat_mul_omp, test_pipeline_run) {
  constexpr size_t kCount = 250;
  constexpr double kInaccuracy = 0.001;
  auto a_matrix = GenerateRandomData(static_cast<int>(kCount * kCount));
  auto single_matrix = GenerateSingleMatrix(static_cast<int>(kCount * kCount));
  std::vector<double> out(kCount * kCount, 0);
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_matrix.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(single_matrix.data()));
  for (int i = 0; i < 4; ++i) {
    task_data_omp->inputs_count.emplace_back(kCount);
  }
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());
  auto test_task_omp = std::make_shared<sarafanov_m_canon_mat_mul_omp::CanonMatMulOMP>(task_data_omp);
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
  for (size_t i = 0; i < kCount * kCount; ++i) {
    EXPECT_NEAR(out[i], a_matrix[i], kInaccuracy);
  }
}

TEST(sarafanov_m_canon_mat_mul_omp, test_task_run) {
  constexpr size_t kCount = 250;
  constexpr double kInaccuracy = 0.001;
  auto a_matrix = GenerateRandomData(static_cast<int>(kCount * kCount));
  auto single_matrix = GenerateSingleMatrix(static_cast<int>(kCount * kCount));
  std::vector<double> out(kCount * kCount, 0);
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_matrix.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(single_matrix.data()));
  for (int i = 0; i < 4; ++i) {
    task_data_omp->inputs_count.emplace_back(kCount);
  }
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  auto test_task_omp = std::make_shared<sarafanov_m_canon_mat_mul_omp::CanonMatMulOMP>(task_data_omp);

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
  for (size_t i = 0; i < kCount * kCount; ++i) {
    EXPECT_NEAR(out[i], a_matrix[i], kInaccuracy);
  }
}
