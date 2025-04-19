#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/komshina_d_image_filtering_vertical_gaussian/include/ops_seq.hpp"

TEST(komshina_d_image_filtering_vertical_gaussian_seq, test_pipeline_run) {
  constexpr int kWidth = 4000;
  constexpr int kHeight = 4000;

  std::vector<unsigned char> in(kWidth * kHeight * 3, 0);
  std::vector<unsigned char> out(kWidth * kHeight * 3, 0);
  std::vector<float> kernel{0.0F, -1.0F, 0.0F, -1.0F, 4.0F, -1.0F, 0.0F, -1.0F, 0.0F};

  for (std::size_t i = 0; i < kHeight; i++) {
    in[(i * kWidth + i) * 3] = 255;
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(kWidth);
  task_data->inputs_count.emplace_back(kHeight);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  auto test_task = std::make_shared<komshina_d_image_filtering_vertical_gaussian_seq::TestTaskSequential>(task_data);

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
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_EQ(in, out);
}

TEST(komshina_d_image_filtering_vertical_gaussian_seq, test_task_run) {
  constexpr int kWidth = 4000;
  constexpr int kHeight = 4000;

  std::vector<unsigned char> in(kWidth * kHeight * 3, 0);
  std::vector<unsigned char> out(kWidth * kHeight * 3, 0);
  std::vector<float> kernel{0.0F, -1.0F, 0.0F, -1.0F, 4.0F, -1.0F, 0.0F, -1.0F, 0.0F};

  for (std::size_t i = 0; i < kHeight; i++) {
    in[(i * kWidth + i) * 3] = 255;
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(kWidth);
  task_data->inputs_count.emplace_back(kHeight);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  auto test_task = std::make_shared<komshina_d_image_filtering_vertical_gaussian_seq::TestTaskSequential>(task_data);

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
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_EQ(in, out);
}