#include "seq/rams_s_vertical_gauss_3x3/include/main.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {
void RunTest(bool pipeline) {
  constexpr int kCount = 4321;

  std::vector<uint8_t> in(kCount * kCount * 3, 0);
  std::vector<uint8_t> out(kCount * kCount * 3, 0);
  std::vector<float> kernel{-1, -1, -1, -1, 3, -1, -1, -1, -1};

  for (std::size_t i = 0; i < kCount; i++) {
    in[(i * kCount + i) * 3] = 1;
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(kCount);
  task_data->inputs_count.emplace_back(kCount);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  auto test_task = std::make_shared<rams_s_vertical_gauss_3x3_seq::TaskSequential>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  if (pipeline) {
    perf_analyzer->PipelineRun(perf_attr, perf_results);
  } else {
    perf_analyzer->TaskRun(perf_attr, perf_results);
  }
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(in, out);
}
}  // namespace

TEST(rams_s_vertical_gauss_3x3_seq, test_pipeline_run) { RunTest(true); }
TEST(rams_s_vertical_gauss_3x3_seq, test_task_run) { RunTest(false); }
