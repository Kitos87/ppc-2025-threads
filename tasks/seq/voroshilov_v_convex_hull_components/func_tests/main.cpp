#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#ifndef _WIN32
#include <opencv2/opencv.hpp>
#endif
#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/voroshilov_v_convex_hull_components/include/chc.hpp"
#include "seq/voroshilov_v_convex_hull_components/include/chc_seq.hpp"

using namespace voroshilov_v_convex_hull_components_seq;

namespace {

bool ValidationTest(int height, int width, std::vector<int>& pixels) {
  int* p_height = &height;
  int* p_width = &width;
  std::vector<int> hulls_indexes_out(height * width);
  std::vector<int> pixels_indexes_out(height * width);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(p_height));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(p_width));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(pixels.data()));
  task_data_seq->inputs_count.emplace_back(pixels.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(hulls_indexes_out.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(pixels_indexes_out.data()));
  task_data_seq->outputs_count.emplace_back(0);

  ChcTaskSequential chc_task_sequential(task_data_seq);
  return chc_task_sequential.ValidationImpl();
}

std::vector<Hull> SimpleRunTest(int height, int width, std::vector<int>& pixels) {
  int* p_height = &height;
  int* p_width = &width;
  std::vector<int> hulls_indexes_out(height * width);
  std::vector<int> pixels_indexes_out(height * width);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(p_height));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(p_width));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(pixels.data()));
  task_data_seq->inputs_count.emplace_back(pixels.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(hulls_indexes_out.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(pixels_indexes_out.data()));
  task_data_seq->outputs_count.emplace_back(0);

  ChcTaskSequential chc_task_sequential(task_data_seq);
  chc_task_sequential.ValidationImpl();
  chc_task_sequential.PreProcessingImpl();
  chc_task_sequential.RunImpl();
  chc_task_sequential.PostProcessingImpl();

  int hulls_size = static_cast<int>(task_data_seq->outputs_count[0]);
  std::vector<Hull> hulls = UnpackHulls(hulls_indexes_out, pixels_indexes_out, height, width, hulls_size);

  return hulls;
}

#ifndef _WIN32

bool ImageRunTest(std::string& src_path, std::string& exp_path) {
  // Load source image:
  cv::Mat src_image = cv::imread(src_path);
  if (src_image.empty()) {
    return false;
  }

  // Convert to shades of gray:
  cv::Mat gray_image;
  cv::cvtColor(src_image, gray_image, cv::COLOR_BGR2GRAY);

  // Convert to black and white:
  cv::Mat bin_image;
  cv::threshold(gray_image, bin_image, 128, 1, cv::THRESH_BINARY);

  // Convert to std::vector<int>:
  std::vector<int> pixels(bin_image.rows * bin_image.cols);
  for (int i = 0; i < bin_image.rows; i++) {
    for (int j = 0; j < bin_image.cols; j++) {
      pixels[(i * bin_image.cols) + j] = bin_image.at<uchar>(i, j);
    }
  }

  int* p_height = &bin_image.rows;
  int height = bin_image.rows;
  int* p_width = &bin_image.cols;
  int width = bin_image.cols;
  std::vector<int> hulls_indexes_out(height * width);
  std::vector<int> pixels_indexes_out(height * width);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(p_height));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(p_width));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(pixels.data()));
  task_data_seq->inputs_count.emplace_back(pixels.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(hulls_indexes_out.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(pixels_indexes_out.data()));
  task_data_seq->outputs_count.emplace_back(0);

  ChcTaskSequential chc_task_sequential(task_data_seq);
  chc_task_sequential.ValidationImpl();
  chc_task_sequential.PreProcessingImpl();
  chc_task_sequential.RunImpl();
  chc_task_sequential.PostProcessingImpl();

  int hulls_size = static_cast<int>(task_data_seq->outputs_count[0]);
  std::vector<Hull> hulls = UnpackHulls(hulls_indexes_out, pixels_indexes_out, height, width, hulls_size);

  // Draw hulls on source image:
  for (Hull hull : hulls) {
    for (size_t i = 0; i < hull.pixels.size() - 1; i++) {
      cv::circle(src_image, cv::Point(hull.pixels[i].x, hull.pixels[i].y), 2, cv::Scalar(0, 0, 255), cv::FILLED);

      cv::line(src_image, cv::Point(hull.pixels[i].x, hull.pixels[i].y),
               cv::Point(hull.pixels[i + 1].x, hull.pixels[i + 1].y), cv::Scalar(0, 0, 255), 1);
    }
    cv::circle(src_image, cv::Point(hull.pixels[hull.pixels.size() - 1].x, hull.pixels[hull.pixels.size() - 1].y), 2,
               cv::Scalar(0, 0, 255), cv::FILLED);

    cv::line(src_image, cv::Point(hull.pixels[hull.pixels.size() - 1].x, hull.pixels[hull.pixels.size() - 1].y),
             cv::Point(hull.pixels[0].x, hull.pixels[0].y), cv::Scalar(0, 0, 255), 1);
  }

  // Load expected image:
  cv::Mat exp_image = cv::imread(exp_path);
  if (exp_image.empty()) {
    src_image.release();
    gray_image.release();
    bin_image.release();
    return false;
  }

  // Compare edited source image with expected image:
  double difference = cv::norm(src_image, exp_image);

  src_image.release();
  gray_image.release();
  bin_image.release();
  exp_image.release();
  // They are same if difference == 0.0
  return difference <= 0.0;
}

#endif

}  // namespace
TEST(voroshilov_v_convex_hull_components_seq, simpleValidationTest) {
  // clang-format off
  std::vector<int> pixels = 
  {0, 1, 0,
   1, 1, 1,
   0, 1, 0};
  // clang-format on
  int height = 0;
  int width = 3;

  ASSERT_FALSE(ValidationTest(height, width, pixels));
}

TEST(voroshilov_v_convex_hull_components_seq, simpleTest0Components) {
  // clang-format off
  std::vector<int> pixels =
  {0, 0, 0,
   0, 0, 0,
   0, 0, 0};
  // clang-format on
  int height = 3;
  int width = 3;
  std::vector<Hull> result_hulls = SimpleRunTest(height, width, pixels);

  ASSERT_TRUE(result_hulls.empty());
}

TEST(voroshilov_v_convex_hull_components_seq, simpleTest1Component) {
  // clang-format off
  std::vector<int> pixels =
  {0, 1, 0,
   1, 1, 1,
   0, 1, 0};
  // clang-format on
  int height = 3;
  int width = 3;
  std::vector<Hull> result_hulls = SimpleRunTest(height, width, pixels);

  std::vector<Hull> expect_hulls;
  Hull hull;
  hull.pixels = {{1, 0}, {0, 1}, {1, 2}, {2, 1}};  // First coordinate is Y, second is X!!!
  expect_hulls.push_back(hull);

  ASSERT_EQ(result_hulls.size(), expect_hulls.size());

  EXPECT_EQ(result_hulls, expect_hulls);
}

TEST(voroshilov_v_convex_hull_components_seq, simpleTest3Components) {
  // clang-format off
  std::vector<int> pixels = 
  {1, 1, 0, 0, 1, 1,
   1, 0, 0, 0, 1, 1,
   0, 0, 1, 0, 0, 0,
   0, 1, 1, 1, 0, 0,
   1, 1, 0, 0, 0, 0};
  // clang-format on
  int height = 5;
  int width = 6;
  std::vector<Hull> result_hulls = SimpleRunTest(height, width, pixels);

  std::vector<Hull> expect_hulls;

  Hull hull1;
  hull1.pixels = {{0, 0}, {0, 1}, {1, 0}};
  expect_hulls.push_back(hull1);

  Hull hull2;
  hull2.pixels = {{0, 4}, {0, 5}, {1, 5}, {1, 4}};
  expect_hulls.push_back(hull2);

  Hull hull3;
  hull3.pixels = {{4, 0}, {2, 2}, {3, 3}, {4, 1}};
  expect_hulls.push_back(hull3);

  ASSERT_EQ(result_hulls.size(), expect_hulls.size());

  EXPECT_EQ(result_hulls, expect_hulls);
}

TEST(voroshilov_v_convex_hull_components_seq, simpleTest5Components) {
  // clang-format off
  std::vector<int> pixels = 
  {1, 1, 0, 0, 0, 1, 1, 0, 0, 0,
   1, 1, 1, 0, 1, 1, 1, 1, 0, 0,
   0, 1, 1, 0, 0, 1, 1, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
   0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
   0, 1, 1, 1, 0, 0, 0, 0, 1, 0,
   0, 0, 1, 0, 0, 1, 0, 1, 1, 0,
   0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
   0, 0, 0, 0, 0, 1, 0, 0, 0, 1};
  // clang-format on
  int height = 11;
  int width = 10;
  std::vector<Hull> result_hulls = SimpleRunTest(height, width, pixels);

  std::vector<Hull> expect_hulls;

  Hull hull1;
  hull1.pixels = {{0, 0}, {0, 1}, {1, 2}, {2, 2}, {2, 1}, {1, 0}};
  expect_hulls.push_back(hull1);

  Hull hull2;
  hull2.pixels = {{1, 4}, {0, 5}, {0, 6}, {1, 7}, {2, 6}, {2, 5}};
  expect_hulls.push_back(hull2);

  Hull hull3;
  hull3.pixels = {{8, 7}, {5, 8}, {6, 9}, {10, 9}};
  expect_hulls.push_back(hull3);

  Hull hull4;
  hull4.pixels = {{7, 1}, {6, 2}, {7, 3}, {8, 2}};
  expect_hulls.push_back(hull4);

  Hull hull5;
  hull5.pixels = {{9, 4}, {8, 5}, {10, 5}};
  expect_hulls.push_back(hull5);

  ASSERT_EQ(result_hulls.size(), expect_hulls.size());

  EXPECT_EQ(result_hulls, expect_hulls);
}

#ifndef _WIN32

TEST(voroshilov_v_convex_hull_components_seq, imageTest0) {
  std::string src_path = ppc::util::GetAbsolutePath("seq/voroshilov_v_convex_hull_components/data/0_image.png");
  std::string exp_path = ppc::util::GetAbsolutePath("seq/voroshilov_v_convex_hull_components/data/0_expected.png");

  ASSERT_TRUE(ImageRunTest(src_path, exp_path));
}

TEST(voroshilov_v_convex_hull_components_seq, imageTest0Incorrect) {
  std::string src_path = ppc::util::GetAbsolutePath("seq/voroshilov_v_convex_hull_components/data/0_image.png");
  std::string inc_path = ppc::util::GetAbsolutePath("seq/voroshilov_v_convex_hull_components/data/0_incorrect.png");

  ASSERT_FALSE(ImageRunTest(src_path, inc_path));
}

TEST(voroshilov_v_convex_hull_components_seq, imageTest1) {
  std::string src_path = ppc::util::GetAbsolutePath("seq/voroshilov_v_convex_hull_components/data/1_image.png");
  std::string exp_path = ppc::util::GetAbsolutePath("seq/voroshilov_v_convex_hull_components/data/1_expected.png");

  ASSERT_TRUE(ImageRunTest(src_path, exp_path));
}

TEST(voroshilov_v_convex_hull_components_seq, imageTest1Incorrect) {
  std::string src_path = ppc::util::GetAbsolutePath("seq/voroshilov_v_convex_hull_components/data/1_image.png");
  std::string inc_path = ppc::util::GetAbsolutePath("seq/voroshilov_v_convex_hull_components/data/1_incorrect.png");

  ASSERT_FALSE(ImageRunTest(src_path, inc_path));
}

TEST(voroshilov_v_convex_hull_components_seq, imageTest2) {
  std::string src_path = ppc::util::GetAbsolutePath("seq/voroshilov_v_convex_hull_components/data/2_image.png");
  std::string exp_path = ppc::util::GetAbsolutePath("seq/voroshilov_v_convex_hull_components/data/2_expected.png");

  ASSERT_TRUE(ImageRunTest(src_path, exp_path));
}

TEST(voroshilov_v_convex_hull_components_seq, imageTest3) {
  std::string src_path = ppc::util::GetAbsolutePath("seq/voroshilov_v_convex_hull_components/data/3_image.png");
  std::string exp_path = ppc::util::GetAbsolutePath("seq/voroshilov_v_convex_hull_components/data/3_expected.png");

  ASSERT_TRUE(ImageRunTest(src_path, exp_path));
}

TEST(voroshilov_v_convex_hull_components_seq, imageTest4) {
  std::string src_path = ppc::util::GetAbsolutePath("seq/voroshilov_v_convex_hull_components/data/4_image.png");
  std::string exp_path = ppc::util::GetAbsolutePath("seq/voroshilov_v_convex_hull_components/data/4_expected.png");

  ASSERT_TRUE(ImageRunTest(src_path, exp_path));
}

TEST(voroshilov_v_convex_hull_components_seq, imageTest5) {
  std::string src_path = ppc::util::GetAbsolutePath("seq/voroshilov_v_convex_hull_components/data/5_image.png");
  std::string exp_path = ppc::util::GetAbsolutePath("seq/voroshilov_v_convex_hull_components/data/5_expected.png");

  ASSERT_TRUE(ImageRunTest(src_path, exp_path));
}

TEST(voroshilov_v_convex_hull_components_seq, imageTest6) {
  std::string src_path = ppc::util::GetAbsolutePath("seq/voroshilov_v_convex_hull_components/data/6_image.png");
  std::string exp_path = ppc::util::GetAbsolutePath("seq/voroshilov_v_convex_hull_components/data/6_expected.png");

  ASSERT_TRUE(ImageRunTest(src_path, exp_path));
}

TEST(voroshilov_v_convex_hull_components_seq, imageTest7) {
  std::string src_path = ppc::util::GetAbsolutePath("seq/voroshilov_v_convex_hull_components/data/7_image.png");
  std::string exp_path = ppc::util::GetAbsolutePath("seq/voroshilov_v_convex_hull_components/data/7_expected.png");

  ASSERT_TRUE(ImageRunTest(src_path, exp_path));
}

#endif
