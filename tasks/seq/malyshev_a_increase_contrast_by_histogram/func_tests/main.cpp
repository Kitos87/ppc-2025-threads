#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#ifndef _WIN32
#include <opencv2/opencv.hpp>
#endif
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/malyshev_a_increase_contrast_by_histogram/include/ops_seq.hpp"

namespace {
std::shared_ptr<ppc::core::TaskData> TestPrepare(std::vector<uint8_t>& input, std::vector<uint8_t>& output) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(input.data());
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(output.data());
  task_data->outputs_count.emplace_back(output.size());
  return task_data;
}

void TestRun(ppc::core::TaskDataPtr& task_data) {
  auto task = std::make_shared<malyshev_a_increase_contrast_by_histogram_seq::TestTaskSequential>(task_data);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
}
}  // namespace

TEST(malyshev_a_increase_contrast_by_histogram_seq, all_pixels_are_equal) {
  constexpr size_t kSize = 10;

  std::vector<uint8_t> input(kSize, 100);
  std::vector<uint8_t> output(kSize, 0);

  auto task_data = TestPrepare(input, output);
  TestRun(task_data);

  EXPECT_TRUE(std::ranges::equal(input, output));
}

TEST(malyshev_a_increase_contrast_by_histogram_seq, black_and_white) {
  constexpr size_t kSize = 10;

  std::vector<uint8_t> input(kSize, 0);
  std::vector<uint8_t> output(kSize, 0);

  for (size_t i = kSize / 2; i < kSize; ++i) {
    input[i] = 255;
  }

  auto task_data = TestPrepare(input, output);
  TestRun(task_data);

  EXPECT_TRUE(std::all_of(output.begin(), output.begin() + kSize / 2, [](uint8_t val) { return val == 0; }));
  EXPECT_TRUE(std::all_of(output.begin() + kSize / 2, output.end(), [](uint8_t val) { return val == 255; }));
}

TEST(malyshev_a_increase_contrast_by_histogram_seq, low_contrast) {
  constexpr size_t kSize = 100;

  std::vector<uint8_t> input(kSize);
  std::vector<uint8_t> output(kSize);

  for (size_t i = 0; i < kSize; ++i) {
    input[i] = 100 + static_cast<uint8_t>(i % 51);
  }

  auto task_data = TestPrepare(input, output);
  TestRun(task_data);

  auto [input_min, input_max] = std::ranges::minmax_element(input);
  auto [output_min, output_max] = std::ranges::minmax_element(output);
  EXPECT_LE(*output_min, *input_min);
  EXPECT_GE(*output_max, *input_max);
}

#ifndef _WIN32
TEST(malyshev_a_increase_contrast_by_histogram_seq, pic_from_file) {
  cv::Mat img = cv::imread(ppc::util::GetAbsolutePath("seq/malyshev_a_increase_contrast_by_histogram/data/input.jpg"),
                           cv::IMREAD_GRAYSCALE);
  cv::Mat reference =
      cv::imread(ppc::util::GetAbsolutePath("seq/malyshev_a_increase_contrast_by_histogram/data/reference.jpg"),
                 cv::IMREAD_GRAYSCALE);

  std::vector<uint8_t> input(img.data, img.data + img.total());
  std::vector<uint8_t> output(input.size());

  auto task_data = TestPrepare(input, output);
  TestRun(task_data);

  auto [input_min, input_max] = std::ranges::minmax_element(input);
  auto [output_min, output_max] = std::ranges::minmax_element(output);
  EXPECT_LE(*output_min, *input_min);
  EXPECT_GE(*output_max, *input_max);

  cv::Mat result(img.rows, img.cols, CV_8UC1, output.data());
  double mse = cv::norm(result, reference, cv::NORM_L2) / (result.rows * result.cols);
  EXPECT_LE(mse, 0.01);
}
#endif

TEST(malyshev_a_increase_contrast_by_histogram_seq, invalid_input) {
  std::vector<uint8_t> input = {1, 2, 3};
  std::vector<uint8_t> output(2);

  auto task_data = TestPrepare(input, output);

  malyshev_a_increase_contrast_by_histogram_seq::TestTaskSequential task(task_data);
  EXPECT_FALSE(task.Validation());
}

TEST(malyshev_a_increase_contrast_by_histogram_seq, empty_input) {
  std::vector<uint8_t> input;
  std::vector<uint8_t> output;

  auto task_data = TestPrepare(input, output);

  malyshev_a_increase_contrast_by_histogram_seq::TestTaskSequential task(task_data);
  EXPECT_FALSE(task.Validation());
}

TEST(malyshev_a_increase_contrast_by_histogram_seq, single_pixel) {
  std::vector<uint8_t> input = {128};
  std::vector<uint8_t> output(1);

  auto task_data = TestPrepare(input, output);
  TestRun(task_data);

  EXPECT_EQ(output[0], input[0]);
}

TEST(malyshev_a_increase_contrast_by_histogram_seq, preserve_order) {
  std::vector<uint8_t> input = {50, 100, 150, 200};
  std::vector<uint8_t> output(4);

  auto task_data = TestPrepare(input, output);
  TestRun(task_data);

  for (size_t i = 0; i < input.size() - 1; ++i) {
    EXPECT_LT(output[i], output[i + 1]);
  }
}
