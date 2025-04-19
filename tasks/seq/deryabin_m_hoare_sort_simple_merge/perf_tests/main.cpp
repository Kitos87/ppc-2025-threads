#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/deryabin_m_hoare_sort_simple_merge/include/ops_seq.hpp"

TEST(deryabin_m_hoare_sort_simple_merge_seq, test_pipeline_run_Seq) {
  // Create data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distribution(-100, 100);
  std::vector<double> input_array(512000);
  std::ranges::generate(input_array.begin(), input_array.end(), [&] { return distribution(gen); });
  std::shuffle(input_array.begin(), input_array.end(), gen);
  std::vector<std::vector<double>> in_array(1, input_array);
  size_t chunk_count = 256;
  std::vector<double> output_array(64000);
  std::vector<std::vector<double>> out_array(1, output_array);
  std::vector<double> true_solution(input_array);
  std::ranges::sort(true_solution.begin(), true_solution.end());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_array.data()));
  task_data_seq->inputs_count.emplace_back(input_array.size());
  task_data_seq->inputs_count.emplace_back(chunk_count);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_array.data()));
  task_data_seq->outputs_count.emplace_back(output_array.size());

  auto hoare_sort_simple_merge_task_sequential =
      std::make_shared<deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(hoare_sort_simple_merge_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(true_solution, out_array[0]);
}

TEST(deryabin_m_hoare_sort_simple_merge_seq, test_task_run_Seq) {
  // Create data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distribution(-100, 100);
  std::vector<double> input_array(512000);
  std::ranges::generate(input_array.begin(), input_array.end(), [&] { return distribution(gen); });
  std::shuffle(input_array.begin(), input_array.end(), gen);
  std::vector<std::vector<double>> in_array(1, input_array);
  size_t chunk_count = 256;
  std::vector<double> output_array(64000);
  std::vector<std::vector<double>> out_array(1, output_array);
  std::vector<double> true_solution(input_array);
  std::ranges::sort(true_solution.begin(), true_solution.end());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_array.data()));
  task_data_seq->inputs_count.emplace_back(input_array.size());
  task_data_seq->inputs_count.emplace_back(chunk_count);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_array.data()));
  task_data_seq->outputs_count.emplace_back(output_array.size());

  auto hoare_sort_simple_merge_task_sequential =
      std::make_shared<deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(hoare_sort_simple_merge_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(true_solution, out_array[0]);
}
