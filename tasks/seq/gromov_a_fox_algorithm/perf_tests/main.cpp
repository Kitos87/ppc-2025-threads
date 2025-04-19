#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/gromov_a_fox_algorithm/include/ops_seq.hpp"

TEST(gromov_a_fox_algorithm_seq, test_pipeline_run) {
  constexpr size_t kN = 400;

  std::vector<double> a(kN * kN, 0.0);
  std::vector<double> b(kN * kN, 0.0);
  std::vector<double> out(kN * kN, 0.0);

  for (size_t i = 0; i < kN; ++i) {
    for (size_t j = 0; j < kN; ++j) {
      a[(i * kN) + j] = static_cast<double>(i + j + 1);
      b[(i * kN) + j] = static_cast<double>(kN - i + j + 1);
    }
  }

  std::vector<double> input;
  input.insert(input.end(), a.begin(), a.end());
  input.insert(input.end(), b.begin(), b.end());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<gromov_a_fox_algorithm_seq::TestTaskSequential>(task_data_seq);

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

  std::vector<double> expected(kN * kN, 0.0);
  for (size_t i = 0; i < kN; ++i) {
    for (size_t j = 0; j < kN; ++j) {
      for (size_t k = 0; k < kN; ++k) {
        expected[(i * kN) + j] += a[(i * kN) + k] * b[(k * kN) + j];
      }
    }
  }

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], expected[i], 1e-9);
  }
}

TEST(gromov_a_fox_algorithm_seq, test_task_run) {
  constexpr size_t kN = 400;

  std::vector<double> a(kN * kN, 0.0);
  std::vector<double> b(kN * kN, 0.0);
  std::vector<double> out(kN * kN, 0.0);

  for (size_t i = 0; i < kN; ++i) {
    for (size_t j = 0; j < kN; ++j) {
      a[(i * kN) + j] = static_cast<double>(i + j + 1);
      b[(i * kN) + j] = static_cast<double>(kN - i + j + 1);
    }
  }

  std::vector<double> input;
  input.insert(input.end(), a.begin(), a.end());
  input.insert(input.end(), b.begin(), b.end());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<gromov_a_fox_algorithm_seq::TestTaskSequential>(task_data_seq);

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

  std::vector<double> expected(kN * kN, 0.0);
  for (size_t i = 0; i < kN; ++i) {
    for (size_t j = 0; j < kN; ++j) {
      for (size_t k = 0; k < kN; ++k) {
        expected[(i * kN) + j] += a[(i * kN) + k] * b[(k * kN) + j];
      }
    }
  }

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], expected[i], 1e-9);
  }
}
