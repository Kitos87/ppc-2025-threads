#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/malyshev_v_radix_sort/include/ops_seq.hpp"

namespace malyshev_v_radix_sort_seq {
namespace {
std::vector<double> GenerateRandomVector(int size, double min_value, double max_value) {
  std::vector<double> random_vector(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(min_value, max_value);
  for (int i = 0; i < size; ++i) {
    random_vector[i] = dis(gen);
  }
  return random_vector;
}
}  // namespace
}  // namespace malyshev_v_radix_sort_seq

TEST(malyshev_v_radix_sort_seq, test_pipeline_run) {
  const int size = 1000000;
  std::vector<double> input_vector = malyshev_v_radix_sort_seq::GenerateRandomVector(size, -1000.0, 1000.0);
  std::vector<double> out(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data_seq->inputs_count.emplace_back(input_vector.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<malyshev_v_radix_sort_seq::RadixSortSequential>(task_data_seq);

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

  std::vector<double> reference = input_vector;
  std::ranges::sort(reference);
  ASSERT_EQ(out, reference);
}

TEST(malyshev_v_radix_sort_seq, test_task_run) {
  const int size = 1000000;
  std::vector<double> input_vector = malyshev_v_radix_sort_seq::GenerateRandomVector(size, -1000.0, 1000.0);
  std::vector<double> out(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data_seq->inputs_count.emplace_back(input_vector.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  auto test_task_sequential = std::make_shared<malyshev_v_radix_sort_seq::RadixSortSequential>(task_data_seq);

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

  std::vector<double> reference = input_vector;
  std::ranges::sort(reference);
  ASSERT_EQ(out, reference);
}