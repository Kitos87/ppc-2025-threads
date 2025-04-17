#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <random>
#include <set>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/alputov_i_graham_scan/include/ops_seq.hpp"

namespace {
void GenerateRandomData(std::vector<alputov_i_graham_scan_seq::Point>& data, size_t count) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-1000.0, 1000.0);

  data.clear();
  data.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    data.emplace_back(dist(gen), dist(gen));
  }
}

void ValidateTask(alputov_i_graham_scan_seq::TestTaskSequential& task) {
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}

std::vector<alputov_i_graham_scan_seq::Point> GenerateStarPoints(size_t num_points_star) {
  std::vector<alputov_i_graham_scan_seq::Point> input;
  for (size_t i = 0; i < num_points_star; ++i) {
    double angle = 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(num_points_star);
    input.emplace_back(20.0 * cos(angle), 20.0 * sin(angle));
    input.emplace_back(5.0 * cos(angle + (std::numbers::pi / static_cast<double>(num_points_star))),
                       5.0 * sin(angle + (std::numbers::pi / static_cast<double>(num_points_star))));
  }
  return input;
}

void AssertConvexHullSize(const std::vector<alputov_i_graham_scan_seq::Point>& convex_hull,
                          const std::vector<alputov_i_graham_scan_seq::Point>& input, size_t expected_min_size) {
  EXPECT_LE(convex_hull.size(), input.size());
  EXPECT_GE(convex_hull.size(), expected_min_size);
}

void AssertStarConvexHullPoints(const std::vector<alputov_i_graham_scan_seq::Point>& convex_hull,
                                size_t num_points_star) {
  std::set<alputov_i_graham_scan_seq::Point> hull_set(convex_hull.begin(), convex_hull.end());
  EXPECT_EQ(hull_set.size(), convex_hull.size());
  EXPECT_EQ(hull_set.size(), num_points_star);
  for (size_t i = 0; i < num_points_star; ++i) {
    double angle = 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(num_points_star);
    EXPECT_TRUE(hull_set.count({20.0 * cos(angle), 20.0 * sin(angle)}));
  }
}

void ValidateStarConvexHull(alputov_i_graham_scan_seq::TestTaskSequential& task,
                            const std::vector<alputov_i_graham_scan_seq::Point>& input, size_t num_points_star) {
  ValidateTask(task);

  EXPECT_FALSE(task.GetConvexHull().empty());
  const auto& convex_hull = task.GetConvexHull();

  AssertConvexHullSize(convex_hull, input, 3U);
  AssertStarConvexHullPoints(convex_hull, num_points_star);
}
}  // namespace

TEST(alputov_i_graham_scan_seq, minimal_triangle_case) {
  std::vector<alputov_i_graham_scan_seq::Point> input = {{0, 0}, {2, 0}, {1, 2}};
  std::vector<alputov_i_graham_scan_seq::Point> output(3);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  alputov_i_graham_scan_seq::TestTaskSequential task(task_data);
  ValidateTask(task);

  EXPECT_EQ(task.GetConvexHull().size(), 3U);
}

TEST(alputov_i_graham_scan_seq, collinear_points) {
  std::vector<alputov_i_graham_scan_seq::Point> input = {{0, 0}, {1, 1}, {2, 2}, {3, 3}};
  std::vector<alputov_i_graham_scan_seq::Point> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  alputov_i_graham_scan_seq::TestTaskSequential task(task_data);
  ValidateTask(task);

  const auto& convex_hull = task.GetConvexHull();
  EXPECT_EQ(convex_hull.size(), 2U);
  bool order1 = (convex_hull[0].x == 0 && convex_hull[1].x == 3);
  bool order2 = (convex_hull[0].x == 3 && convex_hull[1].x == 0);
  EXPECT_TRUE(order1 || order2);
}

TEST(alputov_i_graham_scan_seq, perfect_square_case) {
  std::vector<alputov_i_graham_scan_seq::Point> input = {{0, 0}, {0, 5}, {5, 5}, {5, 0}};
  std::vector<alputov_i_graham_scan_seq::Point> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  alputov_i_graham_scan_seq::TestTaskSequential task(task_data);
  ValidateTask(task);
  EXPECT_FALSE(task.GetConvexHull().empty());
  const auto& convex_hull = task.GetConvexHull();
  EXPECT_EQ(convex_hull.size(), 4U);
  std::set<alputov_i_graham_scan_seq::Point> hull_set(convex_hull.begin(), convex_hull.end());
  for (const auto& p : input) {
    EXPECT_TRUE(hull_set.count(p));
  }
}

TEST(alputov_i_graham_scan_seq, random_1000_points) {
  std::vector<alputov_i_graham_scan_seq::Point> input;
  GenerateRandomData(input, 1000);
  input.insert(input.end(), {{-1500, -1500}, {1500, -1500}, {1500, 1500}, {-1500, 1500}});

  std::vector<alputov_i_graham_scan_seq::Point> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  alputov_i_graham_scan_seq::TestTaskSequential task(task_data);
  ValidateTask(task);

  EXPECT_FALSE(task.GetConvexHull().empty());
  const auto& convex_hull = task.GetConvexHull();
  auto contains = [&](double x, double y) {
    return std::ranges::any_of(convex_hull, [x, y](const auto& p) { return p.x == x && p.y == y; });
  };

  EXPECT_TRUE(contains(-1500, -1500));
  EXPECT_TRUE(contains(1500, 1500));
  EXPECT_LE(convex_hull.size(), input.size());
}

TEST(alputov_i_graham_scan_seq, duplicate_points) {
  std::vector<alputov_i_graham_scan_seq::Point> input(10, {2.5, 3.5});
  input.insert(input.end(), {{0, 0}, {5, 0}, {5, 5}});

  std::vector<alputov_i_graham_scan_seq::Point> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  alputov_i_graham_scan_seq::TestTaskSequential task(task_data);
  ValidateTask(task);

  EXPECT_FALSE(task.GetConvexHull().empty());
  const auto& convex_hull = task.GetConvexHull();
  std::set<alputov_i_graham_scan_seq::Point> unique_hull(convex_hull.begin(), convex_hull.end());
  EXPECT_EQ(unique_hull.size(), 4U);
}
TEST(alputov_i_graham_scan_seq, convex_polygon) {
  std::vector<alputov_i_graham_scan_seq::Point> input = {{0, 0}, {4, 0}, {4, 4}, {2, 6}, {0, 4}};
  std::vector<alputov_i_graham_scan_seq::Point> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  alputov_i_graham_scan_seq::TestTaskSequential task(task_data);
  ValidateTask(task);

  EXPECT_FALSE(task.GetConvexHull().empty());
  const auto& convex_hull = task.GetConvexHull();
  EXPECT_EQ(convex_hull.size(), 5U);
  std::set<alputov_i_graham_scan_seq::Point> hull_set(convex_hull.begin(), convex_hull.end());
  for (const auto& p : input) {
    EXPECT_TRUE(hull_set.count(p));
  }
}

TEST(alputov_i_graham_scan_seq, concave_polygon) {
  std::vector<alputov_i_graham_scan_seq::Point> input = {{0, 0}, {4, 0}, {4, 4}, {2, 2}, {0, 4}};
  std::vector<alputov_i_graham_scan_seq::Point> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  alputov_i_graham_scan_seq::TestTaskSequential task(task_data);
  ValidateTask(task);

  EXPECT_FALSE(task.GetConvexHull().empty());
  const auto& convex_hull = task.GetConvexHull();
  EXPECT_EQ(convex_hull.size(), 4U);
  std::set<alputov_i_graham_scan_seq::Point> hull_set(convex_hull.begin(), convex_hull.end());
  EXPECT_TRUE(hull_set.count({0, 0}));
  EXPECT_TRUE(hull_set.count({4, 0}));
  EXPECT_TRUE(hull_set.count({4, 4}));
  EXPECT_TRUE(hull_set.count({0, 4}));
}

TEST(alputov_i_graham_scan_seq, regular_polygon) {
  std::vector<alputov_i_graham_scan_seq::Point> input = {{0, 0}, {2, 0}, {3, 1}, {2, 2}, {0, 2}, {-1, 1}};
  std::vector<alputov_i_graham_scan_seq::Point> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  alputov_i_graham_scan_seq::TestTaskSequential task(task_data);
  ValidateTask(task);

  EXPECT_FALSE(task.GetConvexHull().empty());
  const auto& convex_hull = task.GetConvexHull();
  EXPECT_EQ(convex_hull.size(), 6U);
  std::set<alputov_i_graham_scan_seq::Point> hull_set(convex_hull.begin(), convex_hull.end());
  for (const auto& p : input) {
    EXPECT_TRUE(hull_set.count(p));
  }
}

TEST(alputov_i_graham_scan_seq, circle_figure) {
  std::vector<alputov_i_graham_scan_seq::Point> input;
  size_t num_points = 36;
  for (size_t i = 0; i < num_points; ++i) {
    double angle = 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(num_points);
    input.emplace_back(10.0 * cos(angle), 10.0 * sin(angle));
  }
  std::vector<alputov_i_graham_scan_seq::Point> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  alputov_i_graham_scan_seq::TestTaskSequential task(task_data);
  ValidateTask(task);

  EXPECT_FALSE(task.GetConvexHull().empty());
  const auto& convex_hull = task.GetConvexHull();
  AssertConvexHullSize(convex_hull, input, 3U);
  std::set<alputov_i_graham_scan_seq::Point> hull_set(convex_hull.begin(), convex_hull.end());
  EXPECT_EQ(hull_set.size(), convex_hull.size());
  EXPECT_EQ(hull_set.size(), input.size());
}

TEST(alputov_i_graham_scan_seq, star_figure) {
  size_t num_points_star = 10;
  std::vector<alputov_i_graham_scan_seq::Point> input = GenerateStarPoints(num_points_star);
  std::vector<alputov_i_graham_scan_seq::Point> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  alputov_i_graham_scan_seq::TestTaskSequential task(task_data);
  ValidateStarConvexHull(task, input, num_points_star);
}

TEST(alputov_i_graham_scan_seq, single_point) {
  std::vector<alputov_i_graham_scan_seq::Point> input = {{0, 0}};
  std::vector<alputov_i_graham_scan_seq::Point> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  alputov_i_graham_scan_seq::TestTaskSequential task(task_data);
  EXPECT_FALSE(task.Validation());
}