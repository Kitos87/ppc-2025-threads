#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/shulpin_i_Jarvis_passage/include/ops_seq.hpp"

namespace {
std::vector<shulpin_i_jarvis_seq::Point> GeneratePointsInCircle(size_t num_points,
                                                                const shulpin_i_jarvis_seq::Point& center,
                                                                double radius) {
  std::vector<shulpin_i_jarvis_seq::Point> points;
  for (size_t i = 0; i < num_points; ++i) {
    double angle = 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(num_points);
    double x = center.x + (radius * std::cos(angle));
    double y = center.y + (radius * std::sin(angle));
    points.emplace_back(x, y);
  }
  return points;
}

void MainTestBody(std::vector<shulpin_i_jarvis_seq::Point>& input, std::vector<shulpin_i_jarvis_seq::Point>& expected) {
  std::vector<shulpin_i_jarvis_seq::Point> result(expected.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data_seq->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  task_data_seq->outputs_count.emplace_back(static_cast<uint32_t>(result.size()));

  shulpin_i_jarvis_seq::JarvisSequential seq_task(task_data_seq);
  ASSERT_EQ(seq_task.Validation(), true);
  seq_task.PreProcessing();
  seq_task.Run();
  seq_task.PostProcessing();

  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(expected[i].x, result[i].x);
    ASSERT_EQ(expected[i].y, result[i].y);
  }
}

void TestBodyFalse(std::vector<shulpin_i_jarvis_seq::Point>& input,
                   std::vector<shulpin_i_jarvis_seq::Point>& expected) {
  std::vector<shulpin_i_jarvis_seq::Point> result(expected.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data_seq->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  task_data_seq->outputs_count.emplace_back(static_cast<uint32_t>(result.size()));

  shulpin_i_jarvis_seq::JarvisSequential seq_task(task_data_seq);
  ASSERT_EQ(seq_task.Validation(), false);
}

void TestBodyRandomCircle(std::vector<shulpin_i_jarvis_seq::Point>& input,
                          std::vector<shulpin_i_jarvis_seq::Point>& expected, size_t& num_points) {
  std::vector<shulpin_i_jarvis_seq::Point> result(expected.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data_seq->inputs_count.emplace_back(static_cast<uint32_t>(input.size()));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  task_data_seq->outputs_count.emplace_back(static_cast<uint32_t>(result.size()));

  shulpin_i_jarvis_seq::JarvisSequential seq_task(task_data_seq);
  ASSERT_EQ(seq_task.Validation(), true);
  seq_task.PreProcessing();
  seq_task.Run();
  seq_task.PostProcessing();

  size_t tmp = num_points >> 1;

  for (size_t i = 0; i < result.size(); ++i) {
    size_t idx = (i < tmp) ? (i + tmp) : (i - tmp);
    EXPECT_EQ(expected[i].x, result[idx].x);
    EXPECT_EQ(expected[i].y, result[idx].y);
  }
}
}  // namespace

TEST(shulpin_i_jarvis_seq, square_with_point) {
  std::vector<shulpin_i_jarvis_seq::Point> input = {{0, 0}, {2, 0}, {2, 2}, {0, 2}, {1, 1}};
  std::vector<shulpin_i_jarvis_seq::Point> expected = {{0, 0}, {2, 0}, {2, 2}, {0, 2}};

  MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_seq, ox_line) {
  std::vector<shulpin_i_jarvis_seq::Point> input = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}};
  std::vector<shulpin_i_jarvis_seq::Point> expected = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}};

  MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_seq, triangle) {
  std::vector<shulpin_i_jarvis_seq::Point> input = {{0, 0}, {3, 0}, {1, 2}};
  std::vector<shulpin_i_jarvis_seq::Point> expected = {{0, 0}, {3, 0}, {1, 2}};

  MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_seq, octagone) {
  std::vector<shulpin_i_jarvis_seq::Point> input = {{1, 0}, {2, 0}, {3, 1}, {3, 2}, {2, 3}, {1, 3}, {0, 2}, {0, 1}};
  std::vector<shulpin_i_jarvis_seq::Point> expected = {{0, 1}, {1, 0}, {2, 0}, {3, 1}, {3, 2}, {2, 3}, {1, 3}, {0, 2}};

  MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_seq, repeated_points) {
  std::vector<shulpin_i_jarvis_seq::Point> input = {{0, 0}, {2, 0}, {2, 2}, {0, 2}, {2, 0}, {0, 0}};
  std::vector<shulpin_i_jarvis_seq::Point> expected = {{0, 0}, {2, 0}, {2, 2}, {0, 2}};

  MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_seq, real_case) {
  std::vector<shulpin_i_jarvis_seq::Point> input = {{1, 1}, {3, 2}, {5, 1}, {4, 3}, {2, 4}, {1, 3}, {3, 3}};
  std::vector<shulpin_i_jarvis_seq::Point> expected = {{1, 1}, {5, 1}, {4, 3}, {2, 4}, {1, 3}};

  MainTestBody(input, expected);
}

TEST(shulpin_i_jarvis_seq, one_point_validation_false) {
  std::vector<shulpin_i_jarvis_seq::Point> input = {{0, 0}};
  std::vector<shulpin_i_jarvis_seq::Point> expected = {{0, 0}};

  TestBodyFalse(input, expected);
}

TEST(shulpin_i_jarvis_seq, three_points_validation_false) {
  std::vector<shulpin_i_jarvis_seq::Point> input = {{1, 1}, {2, 2}};
  std::vector<shulpin_i_jarvis_seq::Point> expected = {{1, 1}, {2, 2}};

  TestBodyFalse(input, expected);
}

TEST(shulpin_i_jarvis_seq, zero_points_validation_false) {
  std::vector<shulpin_i_jarvis_seq::Point> input = {};
  std::vector<shulpin_i_jarvis_seq::Point> expected = {};

  TestBodyFalse(input, expected);
}

TEST(shulpin_i_jarvis_seq, circle_r10_p100) {
  shulpin_i_jarvis_seq::Point center{0, 0};

  double radius = 10.0;
  size_t num_points = 100;

  std::vector<shulpin_i_jarvis_seq::Point> input = GeneratePointsInCircle(num_points, center, radius);
  std::vector<shulpin_i_jarvis_seq::Point> expected = input;

  TestBodyRandomCircle(input, expected, num_points);
}

TEST(shulpin_i_jarvis_seq, circle_r10_p1000) {
  shulpin_i_jarvis_seq::Point center{0, 0};

  double radius = 10.0;
  size_t num_points = 1000;

  std::vector<shulpin_i_jarvis_seq::Point> input = GeneratePointsInCircle(num_points, center, radius);
  std::vector<shulpin_i_jarvis_seq::Point> expected = input;

  TestBodyRandomCircle(input, expected, num_points);
}