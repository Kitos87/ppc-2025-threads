#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#ifndef _WIN32
#include <opencv2/opencv.hpp>
#endif
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "omp/korablev_v_sobel_edges/include/ops_omp.hpp"
#include "omp/korablev_v_sobel_edges/include/ops_seq.hpp"

// #define KORABLEV_V_SOBEL_EDGES_CREATE_FIXTURES

class korablev_v_sobel_edges_test_omp  // NOLINT(readability-identifier-naming)
    : public ::testing::TestWithParam<std::string> {};

#ifndef _WIN32
TEST_P(korablev_v_sobel_edges_test_omp, compare_with_fixture) {
  const auto path = [](const std::string& f) { return "omp/korablev_v_sobel_edges/data/" + f + ".png"; };

  auto&& filename = GetParam();
  const auto in_path = ppc::util::GetAbsolutePath(path("in/" + filename));
  const auto ref_path = ppc::util::GetAbsolutePath(path("ref/" + filename));

  cv::Mat img_in = cv::imread(in_path);
  ASSERT_EQ(img_in.channels(), 3);

  std::vector<uint8_t> in(img_in.reshape(1, static_cast<int>(img_in.total()) * img_in.channels()));
  std::vector<uint8_t> out(in.size());

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {in.data()};
  task_data->inputs_count = {static_cast<uint32_t>(img_in.cols), static_cast<uint32_t>(img_in.rows)};
  task_data->outputs = {out.data()};
  task_data->outputs_count.emplace_back(out.size());

  korablev_v_sobel_edges_omp::TestTask task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
#ifdef KORABLEV_V_SOBEL_EDGES_CREATE_FIXTURES
  cv::Mat out_img(img_in.rows, img_in.cols, CV_8UC3, out.data());
  cv::imwrite(ref_path, out_img);
  FAIL() << "Reference image fixture should've been updated";
#else
  cv::Mat img_ref = cv::imread(ref_path);
  ASSERT_FALSE(img_in.empty()) << "Warning! Chech path: " << in_path;
  std::vector<uint8_t> ref(img_ref.reshape(1, static_cast<int>(img_ref.total()) * img_ref.channels()));
  EXPECT_EQ(out, ref);
#endif
}
// clang-format off
    INSTANTIATE_TEST_SUITE_P(                 // NOLINT(misc-use-anonymous-namespace)
      korablev_v_sobel_edges_test_omp,
      korablev_v_sobel_edges_test_omp,
      ::testing::Values(
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "p6",
        "p7",
        "p8",
        "p9"
    
      )
    );
// clang-format on
#endif

TEST(korablev_v_sobel_edges_test_omp, test_random) {
  const std::size_t height = 1'000;
  const std::size_t width = 100;

  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> dist(0, 255);

  std::vector<uint8_t> in(width * height * 3);
  std::ranges::generate(in.begin(), in.end(), [&] { return dist(gen); });
  std::vector<uint8_t> out(width * height * 3);

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {in.data()};
  task_data->inputs_count = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
  task_data->outputs = {out.data()};
  task_data->outputs_count.emplace_back(out.size());

  korablev_v_sobel_edges_omp::TestTask task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  std::vector<uint8_t> ref(width * height * 3);
  task_data->outputs = {ref.data()};
  korablev_v_sobel_edges_omp::TestTaskSeq task_seq(task_data);
  ASSERT_TRUE(task_seq.Validation());
  task_seq.PreProcessing();
  task_seq.Run();
  task_seq.PostProcessing();

  EXPECT_EQ(out, ref);
}