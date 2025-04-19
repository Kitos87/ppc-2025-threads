#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/kapustin_i_jarv_alg/include/ops_seq.hpp"

namespace {
std::vector<std::pair<int, int>> GenerateRandomPoints(size_t count, int min_x, int max_x, int min_y, int max_y) {
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<int> dist_x(min_x, max_x);
  std::uniform_int_distribution<int> dist_y(min_y, max_y);

  std::vector<std::pair<int, int>> random_points;
  random_points.reserve(count);

  for (size_t i = 0; i < count; ++i) {
    random_points.emplace_back(dist_x(rng), dist_y(rng));
  }

  return random_points;
}
}  // namespace

TEST(KapustinJarvAlgSeqTest, FixedPointsWithRandomNoise) {
  std::vector<std::pair<int, int>> fixed_points = {{-1000, -1000}, {1000, -1000}, {1000, 1000}, {-1000, 1000}};

  auto random_points = GenerateRandomPoints(100, -900, 900, -900, 900);

  std::vector<std::pair<int, int>> input_points = fixed_points;
  input_points.insert(input_points.end(), random_points.begin(), random_points.end());

  std::vector<std::pair<int, int>> output_result(fixed_points.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_seq->inputs_count.emplace_back(input_points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_seq->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_EQ(output_result.size(), fixed_points.size());
  for (size_t i = 0; i < fixed_points.size(); ++i) {
    EXPECT_EQ(output_result[i].first, fixed_points[i].first);
    EXPECT_EQ(output_result[i].second, fixed_points[i].second);
  }
}

TEST(KapustinJarvAlgSeqTest, HexagonWithInnerPoints) {
  std::vector<std::pair<int, int>> input_points = {{0, 0}, {4, 0}, {6, 2}, {4, 4}, {0, 4}, {-2, 2},
                                                   {2, 2}, {3, 1}, {1, 3}, {3, 3}, {2, 1}};

  std::vector<std::pair<int, int>> expected_result = {{-2, 2}, {0, 0}, {4, 0}, {6, 2}, {4, 4}, {0, 4}};

  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_seq->inputs_count.emplace_back(input_points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_seq->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i].first, output_result[i].first);
    EXPECT_EQ(expected_result[i].second, output_result[i].second);
  }
}
TEST(KapustinJarvAlgSeqTest, TriangleWithInnerPoints) {
  std::vector<std::pair<int, int>> input_points = {{0, 0}, {5, 8}, {10, 0}, {5, 4}, {3, 2}, {7, 2}, {5, 6}};

  std::vector<std::pair<int, int>> expected_result = {{0, 0}, {10, 0}, {5, 8}};

  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_seq->inputs_count.emplace_back(input_points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_seq->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i].first, output_result[i].first);
    EXPECT_EQ(expected_result[i].second, output_result[i].second);
  }
}

TEST(KapustinJarvAlgSeqTest, CollinearPoints) {
  std::vector<std::pair<int, int>> input_points = {{0, 0}, {5, 0}, {10, 0}, {15, 0}, {20, 0}};

  std::vector<std::pair<int, int>> expected_result = {{0, 0}, {20, 0}};

  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_seq->inputs_count.emplace_back(input_points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_seq->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i].first, output_result[i].first);
    EXPECT_EQ(expected_result[i].second, output_result[i].second);
  }
}

TEST(KapustinJarvAlgSeqTest, PureTriangle) {
  std::vector<std::pair<int, int>> input_points = {{0, 0}, {5, 8}, {10, 0}};

  std::vector<std::pair<int, int>> expected_result = {{0, 0}, {10, 0}, {5, 8}};

  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_seq->inputs_count.emplace_back(input_points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_seq->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i].first, output_result[i].first);
    EXPECT_EQ(expected_result[i].second, output_result[i].second);
  }
}

TEST(KapustinJarvAlgSeqTest, CollinearPoints2) {
  std::vector<std::pair<int, int>> input_points = {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}};
  std::vector<std::pair<int, int>> expected_result = {{0, 0}, {4, 4}};

  std::vector<std::pair<int, int>> output_result(expected_result.size());
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_seq->inputs_count.emplace_back(input_points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_seq->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_seq::TestTaskSequential test_task_sequential(task_data_seq);

  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_EQ(output_result.size(), expected_result.size());
  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i], output_result[i]);
  }
}

TEST(KapustinJarvAlgSeqTest, SquarePlusOne) {
  std::vector<std::pair<int, int>> input_points = {{0, 0}, {0, 4}, {4, 4}, {4, 0}, {2, 2}};
  std::vector<std::pair<int, int>> expected_result = {{0, 0}, {4, 0}, {4, 4}, {0, 4}};

  std::vector<std::pair<int, int>> output_result(expected_result.size());
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_seq->inputs_count.emplace_back(input_points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_seq->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_seq::TestTaskSequential test_task_sequential(task_data_seq);

  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_EQ(output_result.size(), expected_result.size());
  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i], output_result[i]);
  }
}

TEST(KapustinJarvAlgSeqTest, DuplicatePoints) {
  std::vector<std::pair<int, int>> input_points = {{0, 0}, {0, 5}, {5, 5}, {5, 0}, {0, 5}, {5, 5}, {2, 2}};
  std::vector<std::pair<int, int>> expected_result = {{0, 0}, {5, 0}, {5, 5}, {0, 5}};

  std::vector<std::pair<int, int>> output_result(expected_result.size());
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_seq->inputs_count.emplace_back(input_points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_seq->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_seq::TestTaskSequential test_task_sequential(task_data_seq);

  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_EQ(output_result.size(), expected_result.size());
  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i], output_result[i]);
  }
}

TEST(KapustinJarvAlgSeqTest, SinglePoint) {
  std::vector<std::pair<int, int>> input_points = {{1, 1}};
  std::vector<std::pair<int, int>> expected_result = {{1, 1}};
  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_seq->inputs_count.emplace_back(input_points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_seq->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  ASSERT_EQ(output_result.size(), expected_result.size());
  EXPECT_EQ(output_result[0].first, expected_result[0].first);
  EXPECT_EQ(output_result[0].second, expected_result[0].second);
}

TEST(KapustinJarvAlgSeqTest, TwoPoints) {
  std::vector<std::pair<int, int>> input_points = {{0, 0}, {4, 4}};
  std::vector<std::pair<int, int>> expected_result = {{0, 0}, {4, 4}};
  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_seq->inputs_count.emplace_back(input_points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_seq->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  ASSERT_EQ(output_result.size(), expected_result.size());
  EXPECT_EQ(output_result[0].first, expected_result[0].first);
  EXPECT_EQ(output_result[0].second, expected_result[0].second);
  EXPECT_EQ(output_result[1].first, expected_result[1].first);
  EXPECT_EQ(output_result[1].second, expected_result[1].second);
}

TEST(KapustinJarvAlgSeqTest, Circle) {
  std::vector<std::pair<int, int>> input_points = {{0, 5},  {3, 4},   {4, 3},   {5, 0},  {4, -3}, {3, -4},
                                                   {0, -5}, {-3, -4}, {-4, -3}, {-5, 0}, {-4, 3}, {-3, 4}};

  std::vector<std::pair<int, int>> expected_result = {{-5, 0}, {-4, -3}, {-3, -4}, {0, -5}, {3, -4}, {4, -3},
                                                      {5, 0},  {4, 3},   {3, 4},   {0, 5},  {-3, 4}, {-4, 3}};

  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_seq->inputs_count.emplace_back(input_points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_seq->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i].first, output_result[i].first);
    EXPECT_EQ(expected_result[i].second, output_result[i].second);
  }
}

TEST(KapustinJarvAlgSeqTest, Star4Points) {
  std::vector<std::pair<int, int>> input_points = {{0, 5},   {3, 2},  {5, 0},  {3, -2}, {0, -5},
                                                   {-3, -2}, {-5, 0}, {-3, 2}, {0, 0}};

  std::vector<std::pair<int, int>> expected_result = {{-5, 0}, {0, -5}, {5, 0}, {0, 5}};

  std::vector<std::pair<int, int>> output_result(expected_result.size());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_points.data()));
  task_data_seq->inputs_count.emplace_back(input_points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_result.data()));
  task_data_seq->outputs_count.emplace_back(output_result.size());

  kapustin_i_jarv_alg_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  for (size_t i = 0; i < expected_result.size(); ++i) {
    EXPECT_EQ(expected_result[i].first, output_result[i].first);
    EXPECT_EQ(expected_result[i].second, output_result[i].second);
  }
}