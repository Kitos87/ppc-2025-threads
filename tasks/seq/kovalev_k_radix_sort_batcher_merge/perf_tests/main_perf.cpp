#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/kovalev_k_radix_sort_batcher_merge/include/header.hpp"

const long long int kMinLl = std::numeric_limits<long long>::lowest(), kMaxLl = std::numeric_limits<long long>::max();

TEST(kovalev_k_radix_sort_batcher_merge_seq, test_pipeline_run) {
  const unsigned int length = 1000000;
  std::srand(std::time(nullptr));
  const int alpha = rand();
  std::vector<long long int> in(length, alpha);
  std::vector<long long int> out(length);
  std::shared_ptr<ppc::core::TaskData> task_seq = std::make_shared<ppc::core::TaskData>();
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_seq->inputs_count.emplace_back(in.size());
  task_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_seq->outputs_count.emplace_back(out.size());
  auto test_task_sequential = std::make_shared<kovalev_k_radix_sort_batcher_merge_seq::RadixSortBatcherMerge>(task_seq);
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
  auto *tmp = reinterpret_cast<long long int *>(out.data());
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (tmp[i] != in[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}

TEST(kovalev_k_radix_sort_batcher_merge_seq, test_task_run) {
  const unsigned int length = 9000000;
  std::vector<long long int> in(length);
  std::vector<long long int> out(length);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long long int> dis(kMinLl, kMaxLl);
  std::ranges::generate(in.begin(), in.end(), [&]() { return dis(gen); });
  std::vector<long long int> etalon(in);
  std::shared_ptr<ppc::core::TaskData> task_seq = std::make_shared<ppc::core::TaskData>();
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_seq->inputs_count.emplace_back(in.size());
  task_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_seq->outputs_count.emplace_back(out.size());
  auto test_task_sequential = std::make_shared<kovalev_k_radix_sort_batcher_merge_seq::RadixSortBatcherMerge>(task_seq);
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
  std::ranges::sort(etalon.begin(), etalon.end(), [](long long int a, long long int b) { return a < b; });
  auto *tmp = reinterpret_cast<long long int *>(out.data());
  int count_viol = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (tmp[i] != etalon[i]) {
      count_viol++;
    }
  }
  ASSERT_EQ(count_viol, 0);
}