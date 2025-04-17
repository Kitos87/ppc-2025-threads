#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/filateva_e_simpson/include/ops_seq.hpp"

TEST(filateva_e_simpson_seq, test_x_pow_2) {
  size_t mer = 1;
  size_t steps = 100;
  std::vector<double> a = {1};
  std::vector<double> b = {10};
  std::vector<double> res(1, 0);
  filateva_e_simpson_seq::Func f = [](std::vector<double> x) { return x[0] * x[0]; };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(mer);
  task_data->inputs_count.emplace_back(steps);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);

  filateva_e_simpson_seq::Simpson simpson(task_data);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  filateva_e_simpson_seq::Func integral_f = [](std::vector<double> x) { return x[0] * x[0] * x[0] / 3; };

  ASSERT_NEAR(res[0], integral_f(b) - integral_f(a), 0.01);
}

TEST(filateva_e_simpson_seq, test_x_pow_2_negative) {
  size_t mer = 1;
  size_t steps = 100;
  std::vector<double> a = {-10};
  std::vector<double> b = {10};
  std::vector<double> res(1, 0);
  filateva_e_simpson_seq::Func f = [](std::vector<double> x) { return x[0] * x[0]; };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(mer);
  task_data->inputs_count.emplace_back(steps);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);

  filateva_e_simpson_seq::Simpson simpson(task_data);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  filateva_e_simpson_seq::Func integral_f = [](std::vector<double> x) { return x[0] * x[0] * x[0] / 3; };

  ASSERT_NEAR(res[0], integral_f(b) - integral_f(a), 0.01);
}

TEST(filateva_e_simpson_seq, test_x) {
  size_t mer = 1;
  size_t steps = 100;
  std::vector<double> a = {1};
  std::vector<double> b = {100};
  std::vector<double> res(1, 0);
  filateva_e_simpson_seq::Func f = [](std::vector<double> x) { return x[0]; };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(mer);
  task_data->inputs_count.emplace_back(steps);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);

  filateva_e_simpson_seq::Simpson simpson(task_data);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  filateva_e_simpson_seq::Func integral_f = [](std::vector<double> x) { return x[0] * x[0] / 2; };

  ASSERT_NEAR(res[0], integral_f(b) - integral_f(a), 0.01);
}

TEST(filateva_e_simpson_seq, test_x_pow_3) {
  size_t mer = 1;
  size_t steps = 100;
  std::vector<double> a = {1};
  std::vector<double> b = {100};
  std::vector<double> res(1, 0);
  filateva_e_simpson_seq::Func f = [](std::vector<double> x) { return x[0] * x[0] * x[0]; };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(mer);
  task_data->inputs_count.emplace_back(steps);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);

  filateva_e_simpson_seq::Simpson simpson(task_data);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  filateva_e_simpson_seq::Func integral_f = [](std::vector<double> x) { return std::pow(x[0], 4) / 4; };

  ASSERT_NEAR(res[0], integral_f(b) - integral_f(a), 0.01);
}

TEST(filateva_e_simpson_seq, test_x_del) {
  size_t mer = 1;
  size_t steps = 100;
  std::vector<double> a = {1};
  std::vector<double> b = {10};
  std::vector<double> res(1, 0);
  filateva_e_simpson_seq::Func f = [](std::vector<double> x) { return 1 / x[0]; };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(mer);
  task_data->inputs_count.emplace_back(steps);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);

  filateva_e_simpson_seq::Simpson simpson(task_data);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  filateva_e_simpson_seq::Func integral_f = [](std::vector<double> x) { return std::log(x[0]); };

  ASSERT_NEAR(res[0], integral_f(b) - integral_f(a), 0.01);
}

TEST(filateva_e_simpson_seq, test_x_sin) {
  size_t mer = 1;
  size_t steps = 100;
  std::vector<double> a = {1};
  std::vector<double> b = {std::numbers::pi};
  std::vector<double> res(1, 0);
  filateva_e_simpson_seq::Func f = [](std::vector<double> x) { return std::sin(x[0]); };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(mer);
  task_data->inputs_count.emplace_back(steps);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);

  filateva_e_simpson_seq::Simpson simpson(task_data);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  filateva_e_simpson_seq::Func integral_f = [](std::vector<double> x) { return -std::cos(x[0]); };

  ASSERT_NEAR(res[0], integral_f(b) - integral_f(a), 0.01);
}

TEST(filateva_e_simpson_seq, test_x_cos) {
  size_t mer = 1;
  size_t steps = 100;
  std::vector<double> a = {1};
  std::vector<double> b = {std::numbers::pi / 2};
  std::vector<double> res(1, 0);
  filateva_e_simpson_seq::Func f = [](std::vector<double> x) { return std::cos(x[0]); };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(mer);
  task_data->inputs_count.emplace_back(steps);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);

  filateva_e_simpson_seq::Simpson simpson(task_data);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  filateva_e_simpson_seq::Func integral_f = [](std::vector<double> x) { return std::sin(x[0]); };

  ASSERT_NEAR(res[0], integral_f(b) - integral_f(a), 0.01);
}

TEST(filateva_e_simpson_seq, test_gausa) {
  size_t mer = 1;
  size_t steps = 100;
  std::vector<double> a = {0};
  std::vector<double> b = {1};
  std::vector<double> res(1, 0);
  filateva_e_simpson_seq::Func f = [](std::vector<double> x) { return pow(std::numbers::e, -pow(x[0], 2)); };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(mer);
  task_data->inputs_count.emplace_back(steps);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);

  filateva_e_simpson_seq::Simpson simpson(task_data);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  ASSERT_NEAR(res[0], 0.746824, 0.01);
}

TEST(filateva_e_simpson_seq, test_sum_integral) {
  size_t mer = 1;
  size_t steps = 100;
  std::vector<double> a = {0};
  std::vector<double> b = {10};
  std::vector<double> res(1, 0);
  filateva_e_simpson_seq::Func f = [](std::vector<double> x) { return pow(x[0], 3) + pow(x[0], 2) + x[0]; };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(mer);
  task_data->inputs_count.emplace_back(steps);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);
  filateva_e_simpson_seq::Simpson simpson(task_data);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  filateva_e_simpson_seq::Func integral_f = [](std::vector<double> x) {
    return (pow(x[0], 4) / 4) + (pow(x[0], 3) / 3) + (pow(x[0], 2) / 2);
  };

  ASSERT_NEAR(res[0], integral_f(b) - integral_f(a), 0.01);
}

TEST(filateva_e_simpson_seq, test_error_1) {
  size_t mer = 1;
  size_t steps = 100;
  std::vector<double> a = {10};
  std::vector<double> b = {0};
  std::vector<double> res(1, 0);
  filateva_e_simpson_seq::Func f = [](std::vector<double> x) { return x[0] * x[0]; };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(mer);
  task_data->inputs_count.emplace_back(steps);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);

  filateva_e_simpson_seq::Simpson simpson(task_data);
  ASSERT_FALSE(simpson.Validation());
}

TEST(filateva_e_simpson_seq, test_error_2) {
  size_t mer = 1;
  size_t steps = 101;
  std::vector<double> a = {10};
  std::vector<double> b = {0};
  std::vector<double> res(1, 0);
  filateva_e_simpson_seq::Func f = [](std::vector<double> x) { return x[0] * x[0]; };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(mer);
  task_data->inputs_count.emplace_back(steps);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);

  filateva_e_simpson_seq::Simpson simpson(task_data);
  ASSERT_FALSE(simpson.Validation());
}

TEST(filateva_e_simpson_seq, test_x_y_pow_2) {
  size_t mer = 2;
  size_t steps = 100;
  std::vector<double> a = {0, 0};
  std::vector<double> b = {1, 1};
  std::vector<double> res(1, 0);
  filateva_e_simpson_seq::Func f = [](std::vector<double> param) {
    return (param[0] * param[0]) + (param[1] * param[1]);
  };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(mer);
  task_data->inputs_count.emplace_back(steps);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);

  filateva_e_simpson_seq::Simpson simpson(task_data);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  ASSERT_NEAR(res[0], 0.66666, 0.01);
}

TEST(filateva_e_simpson_seq, test_x_y) {
  size_t mer = 2;
  size_t steps = 100;
  std::vector<double> a = {0, 0};
  std::vector<double> b = {10, 10};
  std::vector<double> res(1, 0);
  filateva_e_simpson_seq::Func f = [](std::vector<double> param) { return param[0] + param[1]; };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(mer);
  task_data->inputs_count.emplace_back(steps);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);

  filateva_e_simpson_seq::Simpson simpson(task_data);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  ASSERT_NEAR(res[0], 1000, 0.01);
}

TEST(filateva_e_simpson_seq, test_sin_x_cos_y) {
  size_t mer = 2;
  size_t steps = 100;
  std::vector<double> a = {0, 0};
  std::vector<double> b = {std::numbers::pi, std::numbers::pi / 2};
  std::vector<double> res(1, 0);
  filateva_e_simpson_seq::Func f = [](std::vector<double> param) { return std::sin(param[0]) * std::cos(param[1]); };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(mer);
  task_data->inputs_count.emplace_back(steps);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);

  filateva_e_simpson_seq::Simpson simpson(task_data);
  ASSERT_TRUE(simpson.Validation());
  simpson.PreProcessing();
  simpson.Run();
  simpson.PostProcessing();

  ASSERT_NEAR(res[0], 2, 0.01);
}
