#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/chizhov_m_trapezoid_method/include/ops_seq.hpp"

TEST(chizhov_m_trapezoid_method_seq, one_variable_squared) {
  int div = 20;
  int dim = 1;
  std::vector<double> limits = {0.0, 5.0};

  std::vector<double> res(1, 0);
  auto f = [](const std::vector<double> &f_val) { return f_val[0] * f_val[0]; };
  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
  task_data_seq->inputs_count.emplace_back(sizeof(div));

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_seq->inputs_count.emplace_back(sizeof(dim));

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_seq->inputs_count.emplace_back(limits.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data_seq->outputs_count.emplace_back(res.size() * sizeof(double));

  chizhov_m_trapezoid_method_seq::TestTaskSequential test_task_sequential(task_data_seq);

  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_NEAR(res[0], 41.66, 0.1);
  delete f_object;
}

TEST(chizhov_m_trapezoid_method_seq, one_variable_cube) {
  int div = 45;
  int dim = 1;
  std::vector<double> limits = {0.0, 5.0};

  std::vector<double> res(1, 0);
  auto f = [](const std::vector<double> &f_val) { return f_val[0] * f_val[0] * f_val[0]; };
  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
  task_data_seq->inputs_count.emplace_back(sizeof(div));

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_seq->inputs_count.emplace_back(sizeof(dim));

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_seq->inputs_count.emplace_back(limits.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data_seq->outputs_count.emplace_back(res.size() * sizeof(double));

  chizhov_m_trapezoid_method_seq::TestTaskSequential test_task_sequential(task_data_seq);

  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_NEAR(res[0], 156.25, 0.1);
  delete f_object;
}

TEST(chizhov_m_trapezoid_method_seq, mul_two_variables) {
  int div = 10;
  int dim = 2;
  std::vector<double> limits = {0.0, 5.0, 0.0, 3.0};

  std::vector<double> res(1, 0);
  auto f = [](const std::vector<double> &f_val) { return f_val[0] * f_val[1]; };
  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
  task_data_seq->inputs_count.emplace_back(sizeof(div));

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_seq->inputs_count.emplace_back(sizeof(dim));

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_seq->inputs_count.emplace_back(limits.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data_seq->outputs_count.emplace_back(res.size() * sizeof(double));

  chizhov_m_trapezoid_method_seq::TestTaskSequential test_task_sequential(task_data_seq);

  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_NEAR(res[0], 56.25, 0.1);
  delete f_object;
}

TEST(chizhov_m_trapezoid_method_seq, sum_two_variables) {
  int div = 10;
  int dim = 2;
  std::vector<double> limits = {0.0, 5.0, 0.0, 3.0};

  std::vector<double> res(1, 0);
  auto f = [](const std::vector<double> &f_val) { return f_val[0] + f_val[1]; };
  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
  task_data_seq->inputs_count.emplace_back(sizeof(div));

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_seq->inputs_count.emplace_back(sizeof(dim));

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_seq->inputs_count.emplace_back(limits.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data_seq->outputs_count.emplace_back(res.size() * sizeof(double));

  chizhov_m_trapezoid_method_seq::TestTaskSequential test_task_sequential(task_data_seq);

  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_NEAR(res[0], 60, 0.1);
  delete f_object;
}

TEST(chizhov_m_trapezoid_method_seq, dif_two_variables) {
  int div = 10;
  int dim = 2;
  std::vector<double> limits = {0.0, 5.0, 0.0, 3.0};

  std::vector<double> res(1, 0);
  auto f = [](const std::vector<double> &f_val) { return f_val[1] - f_val[0]; };
  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
  task_data_seq->inputs_count.emplace_back(sizeof(div));

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_seq->inputs_count.emplace_back(sizeof(dim));

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_seq->inputs_count.emplace_back(limits.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data_seq->outputs_count.emplace_back(res.size() * sizeof(double));

  chizhov_m_trapezoid_method_seq::TestTaskSequential test_task_sequential(task_data_seq);

  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_NEAR(res[0], -15, 0.1);
  delete f_object;
}

TEST(chizhov_m_trapezoid_method_seq, cos_one_variable) {
  int div = 45;
  int dim = 1;
  std::vector<double> limits = {0.0, 5.0};

  std::vector<double> res(1, 0);
  auto f = [](const std::vector<double> &f_val) { return std::cos(f_val[0]); };
  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
  task_data_seq->inputs_count.emplace_back(sizeof(div));

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_seq->inputs_count.emplace_back(sizeof(dim));

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_seq->inputs_count.emplace_back(limits.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data_seq->outputs_count.emplace_back(res.size() * sizeof(double));

  chizhov_m_trapezoid_method_seq::TestTaskSequential test_task_sequential(task_data_seq);

  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_NEAR(res[0], -0.95, 0.1);
  delete f_object;
}

TEST(chizhov_m_trapezoid_method_seq, sin_two_variables) {
  int div = 45;
  int dim = 2;
  std::vector<double> limits = {0.0, 5.0, 0.0, 5.0};

  std::vector<double> res(1, 0);
  auto f = [](const std::vector<double> &f_val) { return std::sin(f_val[0] + f_val[1]); };
  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
  task_data_seq->inputs_count.emplace_back(sizeof(div));

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_seq->inputs_count.emplace_back(sizeof(dim));

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_seq->inputs_count.emplace_back(limits.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data_seq->outputs_count.emplace_back(res.size() * sizeof(double));

  chizhov_m_trapezoid_method_seq::TestTaskSequential test_task_sequential(task_data_seq);

  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_NEAR(res[0], -1.37, 0.1);
  delete f_object;
}

TEST(chizhov_m_trapezoid_method_seq, exp_two_variables) {
  int div = 80;
  int dim = 2;
  std::vector<double> limits = {0.0, 3.0, 0.0, 3.0};

  std::vector<double> res(1, 0);
  auto f = [](const std::vector<double> &f_val) { return std::exp(f_val[0] + f_val[1]); };
  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
  task_data_seq->inputs_count.emplace_back(sizeof(div));

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_seq->inputs_count.emplace_back(sizeof(dim));

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_seq->inputs_count.emplace_back(limits.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data_seq->outputs_count.emplace_back(res.size() * sizeof(double));

  chizhov_m_trapezoid_method_seq::TestTaskSequential test_task_sequential(task_data_seq);

  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_NEAR(res[0], 364.25, 0.1);
  delete f_object;
}

TEST(chizhov_m_trapezoid_method_seq, combine_exp_sin_cos) {
  int div = 90;
  int dim = 2;
  std::vector<double> limits = {0.0, 3.0, 0.0, 3.0};

  std::vector<double> res(1, 0);
  auto f = [](const std::vector<double> &f_val) {
    return std::exp(-f_val[0]) * std::sin(f_val[0]) * std::cos(f_val[1]);
  };
  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
  task_data_seq->inputs_count.emplace_back(sizeof(div));

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_seq->inputs_count.emplace_back(sizeof(dim));

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_seq->inputs_count.emplace_back(limits.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data_seq->outputs_count.emplace_back(res.size() * sizeof(double));

  chizhov_m_trapezoid_method_seq::TestTaskSequential test_task_sequential(task_data_seq);

  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  ASSERT_NEAR(res[0], 0.073, 0.1);
  delete f_object;
}

TEST(chizhov_m_trapezoid_method_seq, invalid_value_dim) {
  int div = 10;
  int dim = -2;
  std::vector<double> limits = {0.0, 5.0, 0.0, 3.0};

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
  task_data_seq->inputs_count.emplace_back(sizeof(div));

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_seq->inputs_count.emplace_back(sizeof(dim));

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_seq->inputs_count.emplace_back(limits.size());

  chizhov_m_trapezoid_method_seq::TestTaskSequential test_task_sequential(task_data_seq);

  ASSERT_FALSE(test_task_sequential.ValidationImpl());
}

TEST(chizhov_m_trapezoid_method_seq, invalid_value_div) {
  int div = -10;
  int dim = 2;
  std::vector<double> limits = {0.0, 5.0, 0.0, 3.0};

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
  task_data_seq->inputs_count.emplace_back(sizeof(div));

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_seq->inputs_count.emplace_back(sizeof(dim));

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_seq->inputs_count.emplace_back(limits.size());

  chizhov_m_trapezoid_method_seq::TestTaskSequential test_task_sequential(task_data_seq);

  ASSERT_FALSE(test_task_sequential.ValidationImpl());
}

TEST(chizhov_m_trapezoid_method_seq, invalid_limit_size) {
  int div = -10;
  int dim = 2;
  std::vector<double> limits = {0.0, 5.0, 0.0};

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
  task_data_seq->inputs_count.emplace_back(sizeof(div));

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_seq->inputs_count.emplace_back(sizeof(dim));

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_seq->inputs_count.emplace_back(limits.size());

  chizhov_m_trapezoid_method_seq::TestTaskSequential test_task_sequential(task_data_seq);

  ASSERT_FALSE(test_task_sequential.ValidationImpl());
}
