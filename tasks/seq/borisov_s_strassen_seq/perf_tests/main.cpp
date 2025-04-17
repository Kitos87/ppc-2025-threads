#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/borisov_s_strassen_seq/include/ops_seq.hpp"

namespace {

void GenerateRandomMatrix(int rows, int cols, std::vector<double>& matrix) {
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  matrix.resize(rows * cols);
  for (auto& value : matrix) {
    value = dist(rng);
  }
}

}  // namespace

TEST(borisov_s_strassen_perf_seq, test_pipeline_run) {
  constexpr int kRowsA = 1024;
  constexpr int kColsA = 512;
  constexpr int kRowsB = 512;
  constexpr int kColsB = 1024;

  std::vector<double> a;
  std::vector<double> b;
  GenerateRandomMatrix(kRowsA, kColsA, a);
  GenerateRandomMatrix(kRowsB, kColsB, b);

  std::vector<double> in_data = {static_cast<double>(kRowsA), static_cast<double>(kColsA), static_cast<double>(kRowsB),
                                 static_cast<double>(kColsB)};
  in_data.insert(in_data.end(), a.begin(), a.end());
  in_data.insert(in_data.end(), b.begin(), b.end());

  size_t output_count = 2 + (kRowsA * kColsB);
  std::vector<double> out(output_count, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data_seq->inputs_count.emplace_back(in_data.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<borisov_s_strassen_seq::SequentialStrassenSeq>(task_data_seq);

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

TEST(borisov_s_strassen_perf_seq, test_task_run) {
  constexpr int kRowsA = 1024;
  constexpr int kColsA = 512;
  constexpr int kRowsB = 512;
  constexpr int kColsB = 1024;

  std::vector<double> a;
  std::vector<double> b;
  GenerateRandomMatrix(kRowsA, kColsA, a);
  GenerateRandomMatrix(kRowsB, kColsB, b);

  std::vector<double> in_data = {static_cast<double>(kRowsA), static_cast<double>(kColsA), static_cast<double>(kRowsB),
                                 static_cast<double>(kColsB)};
  in_data.insert(in_data.end(), a.begin(), a.end());
  in_data.insert(in_data.end(), b.begin(), b.end());

  size_t output_count = 2 + (kRowsA * kColsB);
  std::vector<double> out(output_count, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data_seq->inputs_count.emplace_back(in_data.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<borisov_s_strassen_seq::SequentialStrassenSeq>(task_data_seq);

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
