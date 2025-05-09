#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "all/korovin_n_qsort_batcher/include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {
constexpr int kSize = 7000000;
constexpr int kSeed = 25;

std::vector<int> GenerateRndArray(int size, int seed) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> dist(-1000, 1000);

  std::vector<int> result(size);
  for (auto &elem : result) {
    elem = dist(gen);
  }
  return result;
}
}  // namespace

TEST(korovin_n_qsort_batcher_all, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> in = GenerateRndArray(kSize, kSeed);
  std::vector<int> out(in.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(in.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  auto test_task_all = std::make_shared<korovin_n_qsort_batcher_all::TestTaskALL>(task_data_all);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_TRUE(std::ranges::is_sorted(out));
  }
}

TEST(korovin_n_qsort_batcher_all, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> in = GenerateRndArray(kSize, kSeed);
  std::vector<int> out(in.size());

  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_all->inputs_count.emplace_back(in.size());
  task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_all->outputs_count.emplace_back(out.size());

  auto test_task_all = std::make_shared<korovin_n_qsort_batcher_all::TestTaskALL>(task_data_all);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_TRUE(std::ranges::is_sorted(out));
  }
}
