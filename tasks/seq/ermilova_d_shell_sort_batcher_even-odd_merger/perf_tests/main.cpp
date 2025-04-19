#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/ermilova_d_shell_sort_batcher_even-odd_merger/include/ops_seq.hpp"

namespace {
std::vector<int> GetRandomVector(int size, int upper_border, int lower_border) {
  std::random_device dev;
  std::mt19937 gen(dev());
  if (size <= 0) {
    throw "Incorrect size";
  }
  std::vector<int> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = static_cast<int>(lower_border + (gen() % (upper_border - lower_border + 1)));
  }
  return vec;
}
}  // namespace

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_seq, test_pipeline_run) {
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  const int size = 10000;

  bool is_resersed = false;

  // Create data

  std::vector<int> in = GetRandomVector(size, upper_border_test, lower_border_test);
  std::vector<int> out(in.size(), 0);

  std::vector<int> ref = in;
  std::ranges::sort(ref);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_sequential =
      std::make_shared<ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential>(task_data_seq);

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
  ASSERT_EQ(ref, out);
}

TEST(ermilova_d_shell_sort_batcher_even_odd_merger_seq, test_task_run) {
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  const int size = 10000;

  bool is_resersed = false;

  // Create data
  std::vector<int> in = GetRandomVector(size, upper_border_test, lower_border_test);
  std::vector<int> out(in.size(), 0);

  std::vector<int> ref = in;
  std::ranges::sort(ref);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_sequential =
      std::make_shared<ermilova_d_shell_sort_batcher_even_odd_merger_seq::TestTaskSequential>(task_data_seq);

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
  ASSERT_EQ(ref, out);
}
