#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/lysov_i_matrix_multiplication_Fox_algorithm/include/ops_seq.hpp"
namespace lysov_i_matrix_multiplication_fox_algorithm_seq {
std::vector<double> GetRandomMatrix(size_t size, int min_gen_value, int max_gen_value) {
  std::vector<double> matrix(size * size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(min_gen_value, max_gen_value);

  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      matrix[(i * size) + j] = dist(gen);
    }
  }
  return matrix;
}
}  // namespace lysov_i_matrix_multiplication_fox_algorithm_seq
TEST(lysov_i_matrix_multiplication_Fox_algorithm_seq, test_pipeline_run) {
  // Create data
  size_t n = 512;
  size_t block_size = 128;
  int min_gen_value = -1e3;
  int max_gen_value = 1e3;
  std::vector<double> a =
      lysov_i_matrix_multiplication_fox_algorithm_seq::GetRandomMatrix(n, min_gen_value, max_gen_value);
  std::vector<double> b =
      lysov_i_matrix_multiplication_fox_algorithm_seq::GetRandomMatrix(n, min_gen_value, max_gen_value);
  std::vector<double> c(n * n, 0);
  std::shared_ptr<ppc::core::TaskData> task_data_sequential = std::make_shared<ppc::core::TaskData>();
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block_size));
  task_data_sequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data_sequential->inputs_count.emplace_back(n * n);
  task_data_sequential->inputs_count.emplace_back(n * n);
  task_data_sequential->inputs_count.emplace_back(1);
  task_data_sequential->outputs_count.emplace_back(n * n);

  // Create Task
  auto test_task_sequential =
      std::make_shared<lysov_i_matrix_multiplication_fox_algorithm_seq::TestTaskSequential>(task_data_sequential);

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
}

TEST(lysov_i_matrix_multiplication_Fox_algorithm_seq, test_task_run) {
  size_t n = 400;
  size_t block_size = 100;
  int min_gen_value = -1e3;
  int max_gen_value = 1e3;
  std::vector<double> a =
      lysov_i_matrix_multiplication_fox_algorithm_seq::GetRandomMatrix(n, min_gen_value, max_gen_value);
  std::vector<double> b =
      lysov_i_matrix_multiplication_fox_algorithm_seq::GetRandomMatrix(n, min_gen_value, max_gen_value);
  std::vector<double> c(n * n, 0);

  std::shared_ptr<ppc::core::TaskData> task_data_sequential = std::make_shared<ppc::core::TaskData>();
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_sequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(&block_size));
  task_data_sequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(c.data()));
  task_data_sequential->inputs_count.emplace_back(n * n);
  task_data_sequential->inputs_count.emplace_back(n * n);
  task_data_sequential->inputs_count.emplace_back(1);
  task_data_sequential->outputs_count.emplace_back(n * n);

  // Create Task
  auto test_task_sequential =
      std::make_shared<lysov_i_matrix_multiplication_fox_algorithm_seq::TestTaskSequential>(task_data_sequential);

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
}
