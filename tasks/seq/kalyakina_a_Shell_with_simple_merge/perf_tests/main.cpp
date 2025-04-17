#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/kalyakina_a_Shell_with_simple_merge/include/ops_seq.hpp"

namespace {

std::vector<int> CreateReverseSortedVector(unsigned int size, int left);
bool IsSorted(std::vector<int> &vec);

std::vector<int> CreateReverseSortedVector(unsigned int size, const int left) {
  std::vector<int> result;
  while (size-- != 0) {
    result.push_back(left + (int)size);
  }
  return result;
}

bool IsSorted(std::vector<int> &vec) {
  if (vec.size() < 2) {
    return true;
  }
  for (unsigned int i = 1; i < vec.size(); i++) {
    if (vec[i - 1] > vec[i]) {
      return false;
    }
  }
  return true;
}
}  // namespace

TEST(kalyakina_a_Shell_with_simple_merge_seq, test_pipeline_run) {
  // Create data
  std::vector<int> in = CreateReverseSortedVector(3000000, -1500000);
  std::vector<int> out(in.size());

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_sequential =
      std::make_shared<kalyakina_a_shell_with_simple_merge_seq::ShellSortSequential>(task_data_seq);

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
  ASSERT_TRUE(IsSorted(out));
}

TEST(kalyakina_a_Shell_with_simple_merge_seq, test_task_run) {
  // Create data
  std::vector<int> in = CreateReverseSortedVector(3000000, -1500000);
  std::vector<int> out(in.size());

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_sequential =
      std::make_shared<kalyakina_a_shell_with_simple_merge_seq::ShellSortSequential>(task_data_seq);

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
  ASSERT_TRUE(IsSorted(out));
}
