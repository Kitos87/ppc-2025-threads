#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/titov_s_ImageFilter_HorizGaussian3x3/include/ops_seq.hpp"

TEST(titov_s_image_filter_horiz_gaussian3x3_seq, test_pipeline_run) {
  constexpr size_t kWidth = 8000;
  constexpr size_t kHeight = 8000;
  std::vector<double> input_image(kWidth * kHeight, 0.0);
  std::vector<double> output_image(kWidth * kHeight, 0.0);
  std::vector<int> kernel = {1, 2, 1};

  for (size_t i = 0; i < kHeight; ++i) {
    for (size_t j = 0; j < kWidth; ++j) {
      input_image[(i * kWidth) + j] = static_cast<double>(j) / (kWidth - 1) * 255.0;
    }
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_image.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(kernel.data()));
  task_data_seq->inputs_count.emplace_back(input_image.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_image.data()));
  task_data_seq->outputs_count.emplace_back(output_image.size());

  // Create Task
  auto test_task_sequential =
      std::make_shared<titov_s_image_filter_horiz_gaussian3x3_seq::ImageFilterSequential>(task_data_seq);

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
}

TEST(titov_s_image_filter_horiz_gaussian3x3_seq, test_task_run) {
  constexpr size_t kWidth = 20000;
  constexpr size_t kHeight = 20000;
  std::vector<double> input_image(kWidth * kHeight, 0.0);
  std::vector<double> output_image(kWidth * kHeight, 0.0);
  std::vector<int> kernel = {1, 2, 1};

  for (size_t i = 0; i < kHeight; ++i) {
    for (size_t j = 0; j < kWidth; ++j) {
      input_image[(i * kWidth) + j] = static_cast<double>(j) / (kWidth - 1) * 255.0;
    }
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_image.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(kernel.data()));
  task_data_seq->inputs_count.emplace_back(input_image.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_image.data()));
  task_data_seq->outputs_count.emplace_back(output_image.size());

  // Create Task
  auto test_task_sequential =
      std::make_shared<titov_s_image_filter_horiz_gaussian3x3_seq::ImageFilterSequential>(task_data_seq);

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
}
