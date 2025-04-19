#include <gtest/gtest.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/sharamygina_i_multi_dim_monte_carlo/include/ops_seq.h"

namespace {
std::vector<double> GetBoundaries(double left, double right, unsigned int dimension) {
  std::vector<double> v(dimension * 2);
  for (unsigned int i = 0; i < dimension; i++) {
    v[i * 2] = left;
    v[(i * 2) + 1] = right;
  }
  return v;
}
}  // namespace

TEST(sharamygina_i_multi_dim_monte_carlo_seq, WrongInputCountValidationTest) {
  int iterations = 30000;
  std::vector<double> boundaries = GetBoundaries(0.0, 1.0, 1);
  auto test_function = [](const std::vector<double>& values) {
    assert(values.size() == 2);
    return std::sin(values[0]);
  };
  std::function<double(const std::vector<double>&)> function_ptr = test_function;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(boundaries.data()));
  task_data->inputs_count.emplace_back(boundaries.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
  task_data->inputs_count.emplace_back(3);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&function_ptr));
  task_data->inputs_count.emplace_back(7);

  double result = 0.0;
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(1);

  sharamygina_i_multi_dim_monte_carlo_seq::MultiDimMonteCarloTask test_task(task_data);
  ASSERT_FALSE(test_task.ValidationImpl());
}

TEST(sharamygina_i_multi_dim_monte_carlo_seq, WrongOutputCountValidationTest) {
  int iterations = 30000;
  std::vector<double> boundaries = GetBoundaries(0.0, 1.0, 1);
  auto test_function = [](const std::vector<double>& values) {
    assert(values.size() == 2);
    return std::sin(values[0]);
  };
  std::function<double(const std::vector<double>&)> function_ptr = test_function;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(boundaries.data()));
  task_data->inputs_count.emplace_back(boundaries.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&function_ptr));
  task_data->inputs_count.emplace_back(1);

  double result = 0.0;
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(0);

  sharamygina_i_multi_dim_monte_carlo_seq::MultiDimMonteCarloTask test_task(task_data);
  ASSERT_FALSE(test_task.ValidationImpl());
}

TEST(sharamygina_i_multi_dim_monte_carlo_seq, WrongBoundariesValidationTest) {
  int iterations = 30000;
  std::vector<double> boundaries = GetBoundaries(0.0, 1.0, 1);
  auto test_function = [](const std::vector<double>& values) {
    assert(values.size() == 2);
    return std::sin(values[0]);
  };
  std::function<double(const std::vector<double>&)> function_ptr = test_function;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(boundaries.data()));
  task_data->inputs_count.emplace_back(boundaries.size() - 1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&function_ptr));
  task_data->inputs_count.emplace_back(1);

  double result = 0.0;
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(0);

  sharamygina_i_multi_dim_monte_carlo_seq::MultiDimMonteCarloTask test_task(task_data);
  ASSERT_FALSE(test_task.ValidationImpl());
}

TEST(sharamygina_i_multi_dim_monte_carlo_seq, EmptyOutputValidationTest) {
  int iterations = 30000;
  std::vector<double> boundaries = GetBoundaries(0.0, 1.0, 1);
  auto test_function = [](const std::vector<double>& values) {
    assert(values.size() == 2);
    return std::sin(values[0]);
  };
  std::function<double(const std::vector<double>&)> function_ptr = test_function;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(boundaries.data()));
  task_data->inputs_count.emplace_back(boundaries.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&function_ptr));
  task_data->inputs_count.emplace_back(1);

  task_data->outputs_count.emplace_back(0);

  sharamygina_i_multi_dim_monte_carlo_seq::MultiDimMonteCarloTask test_task(task_data);
  ASSERT_FALSE(test_task.ValidationImpl());
}

TEST(sharamygina_i_multi_dim_monte_carlo_seq, EmptyInputValidationTest) {
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs_count.emplace_back(2);
  task_data->inputs_count.emplace_back(1);
  task_data->inputs_count.emplace_back(1);

  double result = 0.0;
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(0);

  sharamygina_i_multi_dim_monte_carlo_seq::MultiDimMonteCarloTask test_task(task_data);
  ASSERT_FALSE(test_task.ValidationImpl());
}

TEST(sharamygina_i_multi_dim_monte_carlo_seq, 1DSinFunction) {
  int iterations = 30000;
  std::vector<double> boundaries = GetBoundaries(0.0, 1.0, 1);
  auto test_function = [](const std::vector<double>& values) {
    assert(values.size() == 2);
    return std::sin(values[0]);
  };
  std::function<double(const std::vector<double>&)> function_ptr = test_function;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(boundaries.data()));
  task_data->inputs_count.emplace_back(boundaries.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&function_ptr));
  task_data->inputs_count.emplace_back(1);

  double result = 0.0;
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(1);

  sharamygina_i_multi_dim_monte_carlo_seq::MultiDimMonteCarloTask test_task(task_data);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  double expected = 0.45969769413186;
  double tol = 0.03 * expected;
  EXPECT_NEAR(result, expected, tol);
}

TEST(sharamygina_i_multi_dim_monte_carlo_seq, 2DFunction) {
  int iterations = 30000;
  std::vector<double> boundaries = GetBoundaries(0.0, 1.0, 2);
  auto test_function = [](const std::vector<double>& values) {
    assert(values.size() == 2);
    return values[0] + values[1];
  };
  std::function<double(const std::vector<double>&)> function_ptr = test_function;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(boundaries.data()));
  task_data->inputs_count.emplace_back(boundaries.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&function_ptr));
  task_data->inputs_count.emplace_back(1);

  double result = 0.0;
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(1);

  sharamygina_i_multi_dim_monte_carlo_seq::MultiDimMonteCarloTask test_task(task_data);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  double expected = 1;
  double tol = 0.03 * expected;
  EXPECT_NEAR(result, expected, tol);
}

TEST(sharamygina_i_multi_dim_monte_carlo_seq, 3DFunction) {
  int iterations = 30000;
  std::vector<double> boundaries = GetBoundaries(0.0, 3.0, 3);
  auto test_function = [](const std::vector<double>& values) {
    assert(values.size() == 3);
    return 1.0;
  };
  std::function<double(const std::vector<double>&)> function_ptr = test_function;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(boundaries.data()));
  task_data->inputs_count.emplace_back(boundaries.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&function_ptr));
  task_data->inputs_count.emplace_back(1);

  double result = 0.0;
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(1);

  sharamygina_i_multi_dim_monte_carlo_seq::MultiDimMonteCarloTask test_task(task_data);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  double expected = 27;
  double tol = 0.03 * expected;
  EXPECT_NEAR(result, expected, tol);
}

TEST(sharamygina_i_multi_dim_monte_carlo_seq, 4DFunction) {
  int iterations = 30000;
  std::vector<double> boundaries = GetBoundaries(-1.0, 5.0, 4);
  auto test_function = [](const std::vector<double>& values) {
    assert(values.size() == 3);
    return (values[0] * values[3]) + values[2] - (0.3 * values[1]);
  };
  std::function<double(const std::vector<double>&)> function_ptr = test_function;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(boundaries.data()));
  task_data->inputs_count.emplace_back(boundaries.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&function_ptr));
  task_data->inputs_count.emplace_back(1);

  double result = 0.0;
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(1);

  sharamygina_i_multi_dim_monte_carlo_seq::MultiDimMonteCarloTask test_task(task_data);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  double expected = 6998.4;
  double tol = 0.03 * expected;
  EXPECT_NEAR(result, expected, tol);
}

TEST(sharamygina_i_multi_dim_monte_carlo_seq, 2DExpFunction) {
  int iterations = 30000;
  std::vector<double> boundaries = GetBoundaries(1.0, 1.5, 2);
  auto test_function = [](const std::vector<double>& values) {
    assert(values.size() == 2);
    return std::exp(values[0] + values[1]);
  };
  std::function<double(const std::vector<double>&)> function_ptr = test_function;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(boundaries.data()));
  task_data->inputs_count.emplace_back(boundaries.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&function_ptr));
  task_data->inputs_count.emplace_back(1);

  double result = 0.0;
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(1);

  sharamygina_i_multi_dim_monte_carlo_seq::MultiDimMonteCarloTask test_task(task_data);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  double expected = 3.109605100711371;
  double tol = 0.03 * expected;
  EXPECT_NEAR(result, expected, tol);
}

TEST(sharamygina_i_multi_dim_monte_carlo_seq, 10DFunction) {
  int iterations = 30000;
  std::vector<double> boundaries = GetBoundaries(0.0, 1.0, 10);
  auto test_function = [](const std::vector<double>& values) {
    assert(values.size() == 3);
    return 5 + values[0];
  };
  std::function<double(const std::vector<double>&)> function_ptr = test_function;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(boundaries.data()));
  task_data->inputs_count.emplace_back(boundaries.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&function_ptr));
  task_data->inputs_count.emplace_back(1);

  double result = 0.0;
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(1);

  sharamygina_i_multi_dim_monte_carlo_seq::MultiDimMonteCarloTask test_task(task_data);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  double expected = 5.5;
  double tol = 0.03 * expected;
  EXPECT_NEAR(result, expected, tol);
}

TEST(sharamygina_i_multi_dim_monte_carlo_seq, 3DFunctionWithDifferentBoundaries) {
  int iterations = 30000;
  std::vector<double> boundaries = {1.0, 2.0, 1.3, 4.4, 0.5, 0.98};
  auto test_function = [](const std::vector<double>& values) {
    assert(values.size() == 3);
    return 1.0;
  };
  std::function<double(const std::vector<double>&)> function_ptr = test_function;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(boundaries.data()));
  task_data->inputs_count.emplace_back(boundaries.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&function_ptr));
  task_data->inputs_count.emplace_back(1);

  double result = 0.0;
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(1);

  sharamygina_i_multi_dim_monte_carlo_seq::MultiDimMonteCarloTask test_task(task_data);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  double expected = 1.488;
  double tol = 0.03 * expected;
  EXPECT_NEAR(result, expected, tol);
}

TEST(sharamygina_i_multi_dim_monte_carlo_seq, 3DSinFunctionWithDifferentBoundaries) {
  int iterations = 30000;
  std::vector<double> boundaries = {1.0, 2.0, 1.3, 4.4, 0.5, 0.98};
  auto test_function = [](const std::vector<double>& values) {
    assert(values.size() == 3);
    return std::sin(values[0]) + std::cos(values[1]) + std::exp(values[2]);
  };
  std::function<double(const std::vector<double>&)> function_ptr = test_function;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(boundaries.data()));
  task_data->inputs_count.emplace_back(boundaries.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&iterations));
  task_data->inputs_count.emplace_back(1);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&function_ptr));
  task_data->inputs_count.emplace_back(1);

  double result = 0.0;
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.emplace_back(1);

  sharamygina_i_multi_dim_monte_carlo_seq::MultiDimMonteCarloTask test_task(task_data);
  ASSERT_TRUE(test_task.ValidationImpl());
  ASSERT_TRUE(test_task.PreProcessingImpl());
  ASSERT_TRUE(test_task.RunImpl());
  ASSERT_TRUE(test_task.PostProcessingImpl());

  double expected = 3.652697810257515;
  double tol = 0.03 * expected;
  EXPECT_NEAR(result, expected, tol);
}