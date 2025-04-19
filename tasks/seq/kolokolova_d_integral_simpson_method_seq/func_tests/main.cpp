#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/kolokolova_d_integral_simpson_method_seq/include/ops_seq.hpp"

TEST(kolokolova_d_integral_simpson_method_seq, test_easy_func) {
  auto func = [](std::vector<double> vec) { return vec[0] * vec[1]; };
  std::vector<int> step = {2, 2};
  std::vector<int> bord = {2, 4, 3, 6};
  double func_result = 0.0;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
  task_data_seq->inputs_count.emplace_back(step.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
  task_data_seq->inputs_count.emplace_back(bord.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  kolokolova_d_integral_simpson_method_seq::TestTaskSequential test_task_sequential(task_data_seq, func);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  double ans = 81.0;
  double error = 1e-5;
  ASSERT_NEAR(func_result, ans, error);
}

TEST(kolokolova_d_integral_simpson_method_seq, test_func_two_value1) {
  auto func = [](std::vector<double> vec) { return 3 * vec[0] * vec[0] * vec[1] * vec[1]; };
  std::vector<int> step = {10, 10};
  std::vector<int> bord = {4, 6, 3, 6};
  double func_result = 0.0;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
  task_data_seq->inputs_count.emplace_back(step.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
  task_data_seq->inputs_count.emplace_back(bord.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  kolokolova_d_integral_simpson_method_seq::TestTaskSequential test_task_sequential(task_data_seq, func);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  double ans = 9576.0;
  double error = 1e-5;
  ASSERT_NEAR(func_result, ans, error);
}

TEST(kolokolova_d_integral_simpson_method_seq, test_func_two_value2) {
  auto func = [](std::vector<double> vec) { return 4 * vec[0] * 2 * vec[1]; };
  std::vector<int> step = {4, 4};
  std::vector<int> bord = {0, 2, 1, 4};
  double func_result = 0.0;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
  task_data_seq->inputs_count.emplace_back(step.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
  task_data_seq->inputs_count.emplace_back(bord.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  kolokolova_d_integral_simpson_method_seq::TestTaskSequential test_task_sequential(task_data_seq, func);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  double ans = 120.0;
  double error = 1e-5;
  ASSERT_NEAR(func_result, ans, error);
}

TEST(kolokolova_d_integral_simpson_method_seq, test_func_two_value3) {
  auto func = [](std::vector<double> vec) { return (vec[0] * vec[1] / 6) + (2 * vec[0]); };
  std::vector<int> step = {8, 8};
  std::vector<int> bord = {3, 8, 1, 5};
  double func_result = 0.0;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
  task_data_seq->inputs_count.emplace_back(step.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
  task_data_seq->inputs_count.emplace_back(bord.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  kolokolova_d_integral_simpson_method_seq::TestTaskSequential test_task_sequential(task_data_seq, func);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  double ans = 275.0;
  double error = 1e-5;
  ASSERT_NEAR(func_result, ans, error);
}

TEST(kolokolova_d_integral_simpson_method_seq, test_func_three_value1) {
  auto func = [](std::vector<double> vec) { return vec[0] * vec[1] * vec[2]; };
  std::vector<int> step = {20, 20, 20};
  std::vector<int> bord = {0, 2, 3, 6, 4, 8};
  double func_result = 0.0;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
  task_data_seq->inputs_count.emplace_back(step.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
  task_data_seq->inputs_count.emplace_back(bord.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  kolokolova_d_integral_simpson_method_seq::TestTaskSequential test_task_sequential(task_data_seq, func);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  double ans = 662.0;
  double error = 0.1;
  ASSERT_NEAR(func_result, ans, error);
}
TEST(kolokolova_d_integral_simpson_method_seq, test_func_three_value2) {
  auto func = [](std::vector<double> vec) { return (2 * vec[2]) + (vec[1] / 5) + (4 * vec[0]); };
  std::vector<int> step = {30, 30, 30};
  std::vector<int> bord = {0, 1, 0, 3, 0, 2};
  double func_result = 0.0;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
  task_data_seq->inputs_count.emplace_back(step.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
  task_data_seq->inputs_count.emplace_back(bord.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  kolokolova_d_integral_simpson_method_seq::TestTaskSequential test_task_sequential(task_data_seq, func);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  double ans = 25.7;
  double error = 0.1;
  ASSERT_NEAR(func_result, ans, error);
}

TEST(kolokolova_d_integral_simpson_method_seq, test_validation1) {
  auto func = [](std::vector<double> vec) { return vec[0] * vec[1]; };
  std::vector<int> step;
  std::vector<int> bord = {0, 1, 0, 3};
  double func_result = 0.0;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
  task_data_seq->inputs_count.emplace_back(step.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
  task_data_seq->inputs_count.emplace_back(bord.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  kolokolova_d_integral_simpson_method_seq::TestTaskSequential test_task_sequential(task_data_seq, func);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}

TEST(kolokolova_d_integral_simpson_method_seq, test_validation2) {
  auto func = [](std::vector<double> vec) { return vec[0] * vec[1]; };
  std::vector<int> step = {2, 2};
  std::vector<int> bord;
  double func_result = 0.0;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
  task_data_seq->inputs_count.emplace_back(step.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
  task_data_seq->inputs_count.emplace_back(bord.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  kolokolova_d_integral_simpson_method_seq::TestTaskSequential test_task_sequential(task_data_seq, func);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}

TEST(kolokolova_d_integral_simpson_method_seq, test_validation3) {
  auto func = [](std::vector<double> vec) { return vec[0] * vec[1]; };
  std::vector<int> step = {2, 2};
  std::vector<int> bord = {10, 0, 20, 5};
  double func_result = 0.0;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
  task_data_seq->inputs_count.emplace_back(step.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
  task_data_seq->inputs_count.emplace_back(bord.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  kolokolova_d_integral_simpson_method_seq::TestTaskSequential test_task_sequential(task_data_seq, func);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}

TEST(kolokolova_d_integral_simpson_method_seq, test_validation4) {
  auto func = [](std::vector<double> vec) { return vec[0] * vec[1]; };
  std::vector<int> step = {2, 2, 3};
  std::vector<int> bord = {0, 4, 0, 5};
  double func_result = 0.0;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
  task_data_seq->inputs_count.emplace_back(step.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
  task_data_seq->inputs_count.emplace_back(bord.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
  task_data_seq->outputs_count.emplace_back(1);

  // Create Task
  kolokolova_d_integral_simpson_method_seq::TestTaskSequential test_task_sequential(task_data_seq, func);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}

TEST(kolokolova_d_integral_simpson_method_seq, test_difficult_func1) {
  auto func = [](std::vector<double> vec) { return (std::cos(vec[0]) * std::sin(vec[1])); };
  std::vector<int> step = {20, 20};
  std::vector<int> bord = {0, 1, 0, 3};
  double func_result = 0.0;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
  task_data_seq->inputs_count.emplace_back(step.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
  task_data_seq->inputs_count.emplace_back(bord.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
  task_data_seq->outputs_count.emplace_back(1);
  // Create Task
  kolokolova_d_integral_simpson_method_seq::TestTaskSequential test_task_sequential(task_data_seq, func);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  double ans = 1.6745;
  double error = 0.0001;
  ASSERT_NEAR(func_result, ans, error);
}

TEST(kolokolova_d_integral_simpson_method_seq, test_difficult_func2) {
  auto func = [](std::vector<double> vec) { return std::exp(vec[1] + vec[0]) + std::sin(vec[1]) - std::cos(vec[0]); };
  std::vector<int> step = {20, 20};
  std::vector<int> bord = {0, 3, 1, 5};
  double func_result = 0.0;

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
  task_data_seq->inputs_count.emplace_back(step.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
  task_data_seq->inputs_count.emplace_back(bord.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
  task_data_seq->outputs_count.emplace_back(1);
  // Create Task
  kolokolova_d_integral_simpson_method_seq::TestTaskSequential test_task_sequential(task_data_seq, func);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  double ans = 2780.9028;
  double error = 0.0001;
  ASSERT_NEAR(func_result, ans, error);
}