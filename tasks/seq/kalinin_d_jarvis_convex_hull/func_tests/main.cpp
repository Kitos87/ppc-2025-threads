// Copyright 2025 Dmitry Kalinin
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/kalinin_d_jarvis_convex_hull/include/ops_seq.hpp"

namespace {
void Random(std::vector<kalinin_d_jarvis_convex_hull_seq::Point> &points,
            std::vector<kalinin_d_jarvis_convex_hull_seq::Point> &hull,
            std::vector<kalinin_d_jarvis_convex_hull_seq::Point> &res_hull) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  task_data_seq->inputs_count.emplace_back(points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_hull.data()));
  task_data_seq->outputs_count.emplace_back(res_hull.size());

  // Create Task
  kalinin_d_jarvis_convex_hull_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  for (std::size_t i = 0; i < hull.size(); i++) {
    EXPECT_EQ(hull[i].x, res_hull[i].x);
    EXPECT_EQ(hull[i].y, res_hull[i].y);
  }
}

}  // namespace

TEST(kalinin_d_jarvis_convex_hull_seq, Empty_Input) {
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> points = {};
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> hull = {};
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> res_hull(hull.size());
  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  task_data_seq->inputs_count.emplace_back(points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_hull.data()));
  task_data_seq->outputs_count.emplace_back(res_hull.size());

  // Create Task
  kalinin_d_jarvis_convex_hull_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_FALSE(test_task_sequential.ValidationImpl());
}

TEST(kalinin_d_jarvis_convex_hull_seq, Single_Point) {
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> points = {{.x = 0, .y = 0}};
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> hull = {{.x = 0, .y = 0}};
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> res_hull(hull.size());

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  task_data_seq->inputs_count.emplace_back(points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_hull.data()));
  task_data_seq->outputs_count.emplace_back(res_hull.size());

  // Create Task
  kalinin_d_jarvis_convex_hull_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  ASSERT_EQ(res_hull[0], hull[0]);
}

TEST(kalinin_d_jarvis_convex_hull_seq, Two_Points) {
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> points = {{.x = 0, .y = 0}, {.x = 1, .y = 1}};
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> hull = {{.x = 0, .y = 0}, {.x = 1, .y = 1}};
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> res_hull(hull.size());

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  task_data_seq->inputs_count.emplace_back(points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_hull.data()));
  task_data_seq->outputs_count.emplace_back(res_hull.size());

  // Create Task
  kalinin_d_jarvis_convex_hull_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  ASSERT_EQ(res_hull[0], hull[0]);
  ASSERT_EQ(res_hull[1], hull[1]);
}

TEST(kalinin_d_jarvis_convex_hull_seq, Duplicate_Points) {
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> points = {{.x = 0, .y = 0}, {.x = 1, .y = 1}, {.x = 2, .y = 2},
                                                                 {.x = 0, .y = 0}, {.x = 1, .y = 1}, {.x = 2, .y = 2}};
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> hull = {{.x = 0, .y = 0}, {.x = 2, .y = 2}};
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> res_hull(hull.size());

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  task_data_seq->inputs_count.emplace_back(points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_hull.data()));
  task_data_seq->outputs_count.emplace_back(res_hull.size());
  // Create Task
  kalinin_d_jarvis_convex_hull_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  ASSERT_EQ(res_hull[0], hull[0]);
  ASSERT_EQ(res_hull[1], hull[1]);
}

TEST(kalinin_d_jarvis_convex_hull_seq, Random_Points) {
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> points = {{.x = 1, .y = 4}, {.x = 3, .y = 8}, {.x = 8, .y = 2},
                                                                 {.x = 5, .y = 5}, {.x = 9, .y = 1}, {.x = 4, .y = 7}};
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> hull = {
      {.x = 1, .y = 4}, {.x = 3, .y = 8}, {.x = 4, .y = 7}, {.x = 9, .y = 1}};
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> res_hull(hull.size());
  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  task_data_seq->inputs_count.emplace_back(points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_hull.data()));
  task_data_seq->outputs_count.emplace_back(res_hull.size());

  // Create Task
  kalinin_d_jarvis_convex_hull_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  for (size_t i = 0; i < hull.size(); ++i) {
    ASSERT_EQ(res_hull[i], hull[i]);
  }
}

TEST(kalinin_d_jarvis_convex_hull_seq, Rectangle_Points) {
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> points = {{.x = 0, .y = 0}, {.x = 0, .y = 5}, {.x = 5, .y = 5},
                                                                 {.x = 5, .y = 0}, {.x = 1, .y = 1}, {.x = 1, .y = 4},
                                                                 {.x = 4, .y = 4}, {.x = 4, .y = 1}};
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> hull = {
      {.x = 0, .y = 0}, {.x = 0, .y = 5}, {.x = 5, .y = 5}, {.x = 5, .y = 0}};
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> res_hull(hull.size());

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  task_data_seq->inputs_count.emplace_back(points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_hull.data()));
  task_data_seq->outputs_count.emplace_back(res_hull.size());

  // Create Task
  kalinin_d_jarvis_convex_hull_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  for (size_t i = 0; i < hull.size(); ++i) {
    ASSERT_EQ(res_hull[i], hull[i]);
  }
}

TEST(kalinin_d_jarvis_convex_hull_seq, Circle_Points) {
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> points = {
      {.x = 0, .y = 1},   {.x = 1, .y = 2},  {.x = 2, .y = 1}, {.x = 1, .y = 0}, {.x = 0, .y = -1}, {.x = -1, .y = -2},
      {.x = -2, .y = -1}, {.x = -1, .y = 0}, {.x = 0, .y = 2}, {.x = 2, .y = 0}, {.x = -2, .y = 0}, {.x = 0, .y = -2}};
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> hull = {{.x = -2, .y = -1}, {.x = -2, .y = 0}, {.x = 0, .y = 2},
                                                               {.x = 1, .y = 2},   {.x = 2, .y = 1},  {.x = 2, .y = 0},
                                                               {.x = 0, .y = -2},  {.x = -1, .y = -2}};

  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> res_hull(hull.size());

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  task_data_seq->inputs_count.emplace_back(points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_hull.data()));
  task_data_seq->outputs_count.emplace_back(res_hull.size());

  // Create Task
  kalinin_d_jarvis_convex_hull_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  for (size_t i = 0; i < hull.size(); ++i) {
    ASSERT_EQ(res_hull[i], hull[i]);
  }
}

TEST(kalinin_d_jarvis_convex_hull_seq, Star_Points) {
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> points = {
      {.x = 0, .y = 3}, {.x = 1, .y = 1}, {.x = 2, .y = 3}, {.x = 3, .y = 1}, {.x = 4, .y = 3},
      {.x = 2, .y = 0}, {.x = 0, .y = 0}, {.x = 4, .y = 0}, {.x = 2, .y = 4}, {.x = 2, .y = -1}};
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> hull = {{.x = 0, .y = 0}, {.x = 0, .y = 3}, {.x = 2, .y = 4},
                                                               {.x = 4, .y = 3}, {.x = 4, .y = 0}, {.x = 2, .y = -1}};

  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> res_hull(hull.size());

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  task_data_seq->inputs_count.emplace_back(points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_hull.data()));
  task_data_seq->outputs_count.emplace_back(res_hull.size());

  // Create Task
  kalinin_d_jarvis_convex_hull_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  for (size_t i = 0; i < hull.size(); ++i) {
    ASSERT_EQ(res_hull[i], hull[i]);
  }
}

TEST(kalinin_d_jarvis_convex_hull_seq, Random_Points_RNG) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<int> dist(-8, 8);

  // Create data
  auto gen = [&dist, &rng]() { return kalinin_d_jarvis_convex_hull_seq::Point{.x = dist(rng), .y = dist(rng)}; };

  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> points(30);
  std::ranges::generate(points.begin(), points.end(), gen);
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> hull = {
      {.x = -19, .y = 9}, {.x = -9, .y = 13},   {.x = 7, .y = 12},    {.x = 14, .y = 5}, {.x = 15, .y = -13},
      {.x = 0, .y = -18}, {.x = -11, .y = -21}, {.x = -14, .y = -14}, {.x = -18, .y = 0}};
  points.insert(points.end(), hull.begin(), hull.end());
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> res_hull(hull.size());
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(points.begin(), points.end(), g);

  Random(points, hull, res_hull);
}
