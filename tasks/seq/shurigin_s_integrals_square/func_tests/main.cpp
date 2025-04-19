#define USE_MATH_DEFINES
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <numbers>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/shurigin_s_integrals_square/include/ops_seq.hpp"

namespace shurigin_s_integrals_square_seq_test {

TEST(shurigin_s_integrals_square_seq, test_integration_x_squared) {
  const double lower_bound = 0.0;
  const double upper_bound = 1.0;
  const int intervals = 1000;
  const double expected_value = 1.0 / 3.0;

  std::vector<double> input_data = {lower_bound, upper_bound, static_cast<double>(intervals)};
  double output_data = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(input_data.size() * sizeof(double));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&output_data));
  task_data->outputs_count.push_back(sizeof(double));

  shurigin_s_integrals_square_seq::Integral integral_task(task_data);
  integral_task.SetFunction([](double x) { return x * x; });

  ASSERT_TRUE(integral_task.PreProcessingImpl());
  ASSERT_TRUE(integral_task.ValidationImpl());
  ASSERT_TRUE(integral_task.RunImpl());
  ASSERT_TRUE(integral_task.PostProcessingImpl());

  ASSERT_NEAR(output_data, expected_value, 1e-3);
}

TEST(shurigin_s_integrals_square_seq, test_integration_linear) {
  const double lower_bound = 0.0;
  const double upper_bound = 1.0;
  const int intervals = 1000;
  const double expected_value = 0.5;

  std::vector<double> input_data = {lower_bound, upper_bound, static_cast<double>(intervals)};
  double output_data = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(input_data.size() * sizeof(double));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&output_data));
  task_data->outputs_count.push_back(sizeof(double));

  shurigin_s_integrals_square_seq::Integral integral_task(task_data);
  integral_task.SetFunction([](double x) { return x; });

  ASSERT_TRUE(integral_task.PreProcessingImpl());
  ASSERT_TRUE(integral_task.ValidationImpl());
  ASSERT_TRUE(integral_task.RunImpl());
  ASSERT_TRUE(integral_task.PostProcessingImpl());

  ASSERT_NEAR(output_data, expected_value, 1e-3);
}

TEST(shurigin_s_integrals_square_seq, test_integration_sine) {
  const double lower_bound = 0.0;
  const double upper_bound = std::numbers::pi;
  const int intervals = 1000;
  const double expected_value = 2.0;

  std::vector<double> input_data = {lower_bound, upper_bound, static_cast<double>(intervals)};
  double output_data = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(input_data.size() * sizeof(double));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&output_data));
  task_data->outputs_count.push_back(sizeof(double));

  shurigin_s_integrals_square_seq::Integral integral_task(task_data);
  integral_task.SetFunction([](double x) { return sin(x); });

  ASSERT_TRUE(integral_task.PreProcessingImpl());
  ASSERT_TRUE(integral_task.ValidationImpl());
  ASSERT_TRUE(integral_task.RunImpl());
  ASSERT_TRUE(integral_task.PostProcessingImpl());

  ASSERT_NEAR(output_data, expected_value, 1e-3);
}

TEST(shurigin_s_integrals_square_seq, test_integration_exponential) {
  const double lower_bound = 0.0;
  const double upper_bound = 1.0;
  const int intervals = 1000;
  const double expected_value = std::numbers::e - 1.0;

  std::vector<double> input_data = {lower_bound, upper_bound, static_cast<double>(intervals)};
  double output_data = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(input_data.size() * sizeof(double));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&output_data));
  task_data->outputs_count.push_back(sizeof(double));

  shurigin_s_integrals_square_seq::Integral integral_task(task_data);
  integral_task.SetFunction([](double x) { return std::exp(x); });

  ASSERT_TRUE(integral_task.PreProcessingImpl());
  ASSERT_TRUE(integral_task.ValidationImpl());
  ASSERT_TRUE(integral_task.RunImpl());
  ASSERT_TRUE(integral_task.PostProcessingImpl());

  ASSERT_NEAR(output_data, expected_value, 1e-3);
}

TEST(shurigin_s_integrals_square_seq, test_function_assignment) {
  std::vector<double> input_data = {0.0, 1.0, 1000};
  double output_data = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(input_data.size() * sizeof(double));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&output_data));
  task_data->outputs_count.push_back(sizeof(double));

  shurigin_s_integrals_square_seq::Integral integral_task(task_data);
  std::function<double(double)> func = [](double x) { return x * x; };
  integral_task.SetFunction(func);

  double test_value = 2.0;
  double expected_value = 4.0;

  ASSERT_EQ(func(test_value), expected_value);
}

TEST(shurigin_s_integrals_square_seq, test_integration_cosine) {
  const double lower_bound = 0.0;
  const double upper_bound = std::numbers::pi / 2;
  const int intervals = 1000;
  const double expected_value = 1.0;

  std::vector<double> input_data = {lower_bound, upper_bound, static_cast<double>(intervals)};
  double output_data = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(input_data.size() * sizeof(double));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&output_data));
  task_data->outputs_count.push_back(sizeof(double));

  shurigin_s_integrals_square_seq::Integral integral_task(task_data);
  integral_task.SetFunction([](double x) { return cos(x); });

  ASSERT_TRUE(integral_task.PreProcessingImpl());
  ASSERT_TRUE(integral_task.ValidationImpl());
  ASSERT_TRUE(integral_task.RunImpl());
  ASSERT_TRUE(integral_task.PostProcessingImpl());

  ASSERT_NEAR(output_data, expected_value, 1e-3);
}

TEST(shurigin_s_integrals_square_seq, test_integration_logarithm) {
  const double lower_bound = 1.0;
  const double upper_bound = 2.0;
  const int intervals = 1000;
  const double expected_value = (2 * std::numbers::ln2) - 1;

  std::vector<double> input_data = {lower_bound, upper_bound, static_cast<double>(intervals)};
  double output_data = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(input_data.size() * sizeof(double));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&output_data));
  task_data->outputs_count.push_back(sizeof(double));

  shurigin_s_integrals_square_seq::Integral integral_task(task_data);
  integral_task.SetFunction([](double x) { return std::log(x); });

  ASSERT_TRUE(integral_task.PreProcessingImpl());
  ASSERT_TRUE(integral_task.ValidationImpl());
  ASSERT_TRUE(integral_task.RunImpl());
  ASSERT_TRUE(integral_task.PostProcessingImpl());

  ASSERT_NEAR(output_data, expected_value, 1e-3);
}

TEST(shurigin_s_integrals_square_seq, test_integration_reciprocal) {
  const double lower_bound = 1.0;
  const double upper_bound = 2.0;
  const int intervals = 1000;
  const double expected_value = std::numbers::ln2;

  std::vector<double> input_data = {lower_bound, upper_bound, static_cast<double>(intervals)};
  double output_data = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(input_data.size() * sizeof(double));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&output_data));
  task_data->outputs_count.push_back(sizeof(double));

  shurigin_s_integrals_square_seq::Integral integral_task(task_data);
  integral_task.SetFunction([](double x) { return 1.0 / x; });

  ASSERT_TRUE(integral_task.PreProcessingImpl());
  ASSERT_TRUE(integral_task.ValidationImpl());
  ASSERT_TRUE(integral_task.RunImpl());
  ASSERT_TRUE(integral_task.PostProcessingImpl());

  ASSERT_NEAR(output_data, expected_value, 1e-3);
}

TEST(shurigin_s_integrals_square_seq, test_integration_sqrt) {
  const double lower_bound = 0.0;
  const double upper_bound = 1.0;
  const int intervals = 1000;
  const double expected_value = 2.0 / 3.0;

  std::vector<double> input_data = {lower_bound, upper_bound, static_cast<double>(intervals)};
  double output_data = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(input_data.size() * sizeof(double));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&output_data));
  task_data->outputs_count.push_back(sizeof(double));

  shurigin_s_integrals_square_seq::Integral integral_task(task_data);
  integral_task.SetFunction([](double x) { return std::sqrt(x); });

  ASSERT_TRUE(integral_task.PreProcessingImpl());
  ASSERT_TRUE(integral_task.ValidationImpl());
  ASSERT_TRUE(integral_task.RunImpl());
  ASSERT_TRUE(integral_task.PostProcessingImpl());

  ASSERT_NEAR(output_data, expected_value, 1e-3);
}

TEST(shurigin_s_integrals_square_seq, test_integration_2d) {
  const double lower_bound_x = 0.0;
  const double upper_bound_x = 1.0;
  const double lower_bound_y = 0.0;
  const double upper_bound_y = 1.0;
  const int intervals_x = 100;
  const int intervals_y = 100;
  const double expected_value = 1.0 / 4.0;

  std::vector<double> input_data = {lower_bound_x,
                                    lower_bound_y,
                                    upper_bound_x,
                                    upper_bound_y,
                                    static_cast<double>(intervals_x),
                                    static_cast<double>(intervals_y)};
  double output_data = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(input_data.size() * sizeof(double));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&output_data));
  task_data->outputs_count.push_back(sizeof(double));

  shurigin_s_integrals_square_seq::Integral integral_task(task_data);
  integral_task.SetFunction([](const std::vector<double>& point) { return point[0] * point[1]; }, 2);

  ASSERT_TRUE(integral_task.PreProcessingImpl());
  ASSERT_TRUE(integral_task.ValidationImpl());
  ASSERT_TRUE(integral_task.RunImpl());
  ASSERT_TRUE(integral_task.PostProcessingImpl());
  ASSERT_NEAR(output_data, expected_value, 1e-3);
}
TEST(shurigin_s_integrals_square_seq, test_integration_2d_square_sum) {
  const double lower_bound_x = 0.0;
  const double upper_bound_x = 1.0;
  const double lower_bound_y = 0.0;
  const double upper_bound_y = 1.0;
  const int intervals_x = 100;
  const int intervals_y = 100;
  const double expected_value = 2.0 / 3.0;

  std::vector<double> input_data = {lower_bound_x,
                                    lower_bound_y,
                                    upper_bound_x,
                                    upper_bound_y,
                                    static_cast<double>(intervals_x),
                                    static_cast<double>(intervals_y)};
  double output_data = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(input_data.size() * sizeof(double));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&output_data));
  task_data->outputs_count.push_back(sizeof(double));

  shurigin_s_integrals_square_seq::Integral integral_task(task_data);
  integral_task.SetFunction(
      [](const std::vector<double>& point) { return (point[0] * point[0]) + (point[1] * point[1]); }, 2);

  ASSERT_TRUE(integral_task.PreProcessingImpl());
  ASSERT_TRUE(integral_task.ValidationImpl());
  ASSERT_TRUE(integral_task.RunImpl());
  ASSERT_TRUE(integral_task.PostProcessingImpl());

  ASSERT_NEAR(output_data, expected_value, 1e-3);
}

TEST(shurigin_s_integrals_square_seq, test_integration_2d_sin_sum) {
  const double lower_bound_x = 0.0;
  const double upper_bound_x = std::numbers::pi / 2;
  const double lower_bound_y = 0.0;
  const double upper_bound_y = std::numbers::pi / 2;
  const int intervals_x = 100;
  const int intervals_y = 100;
  const double expected_value = 2.0;

  std::vector<double> input_data = {lower_bound_x,
                                    lower_bound_y,
                                    upper_bound_x,
                                    upper_bound_y,
                                    static_cast<double>(intervals_x),
                                    static_cast<double>(intervals_y)};
  double output_data = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(input_data.size() * sizeof(double));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&output_data));
  task_data->outputs_count.push_back(sizeof(double));

  shurigin_s_integrals_square_seq::Integral integral_task(task_data);
  integral_task.SetFunction([](const std::vector<double>& point) { return std::sin(point[0] + point[1]); }, 2);

  ASSERT_TRUE(integral_task.PreProcessingImpl());
  ASSERT_TRUE(integral_task.ValidationImpl());
  ASSERT_TRUE(integral_task.RunImpl());
  ASSERT_TRUE(integral_task.PostProcessingImpl());

  ASSERT_NEAR(output_data, expected_value, 1e-3);
}

TEST(shurigin_s_integrals_square_seq, test_integration_3d_product) {
  const double lower_bound_x = 0.0;
  const double upper_bound_x = 1.0;
  const double lower_bound_y = 0.0;
  const double upper_bound_y = 1.0;
  const double lower_bound_z = 0.0;
  const double upper_bound_z = 1.0;
  const int intervals_x = 50;
  const int intervals_y = 50;
  const int intervals_z = 50;
  const double expected_value = 0.125;

  std::vector<double> input_data = {lower_bound_x,
                                    lower_bound_y,
                                    lower_bound_z,
                                    upper_bound_x,
                                    upper_bound_y,
                                    upper_bound_z,
                                    static_cast<double>(intervals_x),
                                    static_cast<double>(intervals_y),
                                    static_cast<double>(intervals_z)};
  double output_data = 0.0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(input_data.size() * sizeof(double));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&output_data));
  task_data->outputs_count.push_back(sizeof(double));

  shurigin_s_integrals_square_seq::Integral integral_task(task_data);
  integral_task.SetFunction([](const std::vector<double>& point) { return point[0] * point[1] * point[2]; }, 3);

  ASSERT_TRUE(integral_task.PreProcessingImpl());
  ASSERT_TRUE(integral_task.ValidationImpl());
  ASSERT_TRUE(integral_task.RunImpl());
  ASSERT_TRUE(integral_task.PostProcessingImpl());

  ASSERT_NEAR(output_data, expected_value, 1e-3);
}
}  // namespace shurigin_s_integrals_square_seq_test