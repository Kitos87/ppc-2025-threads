#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/poroshin_v_multi_integral_with_trapez_method/include/ops_seq.hpp"

namespace {
double Area(std::vector<double> &arguments) { return 1.0 + (arguments.at(0) * 0.0); }
double F1(std::vector<double> &arguments) { return arguments.at(0); }
double F1cos(std::vector<double> &arguments) { return cos(arguments.at(0)); }
double F1Euler(std::vector<double> &arguments) { return 2 * cos(arguments.at(0)) * sin(arguments.at(0)); }
double F2(std::vector<double> &arguments) { return arguments.at(0) * arguments.at(1); }
double F2advanced(std::vector<double> &arguments) { return std::tan(arguments.at(0)) * std::atan(arguments.at(1)); }
double F3(std::vector<double> &arguments) { return arguments.at(0) * arguments.at(1) * arguments.at(2); }
double F3advanced(std::vector<double> &arguments) {
  return sin(arguments.at(0)) * tan(arguments.at(1)) * log(arguments.at(2));
}
double F4(std::vector<double> &arguments) {
  return arguments.at(0) * arguments.at(1) * arguments.at(2) * arguments.at(3);
}
double F4advanced(std::vector<double> &arguments) {
  return (arguments.at(0) * arguments.at(0)) + (arguments.at(1) * arguments.at(1)) +
         (arguments.at(2) * arguments.at(2)) + (arguments.at(3) * arguments.at(3));
}
}  // namespace

TEST(poroshin_v_multi_integral_with_trapez_method_seq, invalid_size) {
  std::vector<int> n;
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> out;
  std::shared_ptr<ppc::core::TaskData> task_seq = std::make_shared<ppc::core::TaskData>();
  task_seq->inputs_count.emplace_back(n.size());
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_seq->outputs_count.emplace_back(out.size());
  poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential tmp_task_seq(task_seq, F1);
  ASSERT_FALSE(tmp_task_seq.ValidationImpl());
}

TEST(poroshin_v_multi_integral_with_trapez_method_seq, invalid_out) {
  size_t dim = 10;
  std::vector<int> n(dim);
  std::vector<double> a(dim);
  std::vector<double> b(dim);
  std::vector<double> out(2);
  std::shared_ptr<ppc::core::TaskData> task_seq = std::make_shared<ppc::core::TaskData>();
  task_seq->inputs_count.emplace_back(n.size());
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_seq->outputs_count.emplace_back(out.size());
  poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential tmp_task_seq(task_seq, F1);
  ASSERT_FALSE(tmp_task_seq.ValidationImpl());
}

TEST(poroshin_v_multi_integral_with_trapez_method_seq, minus_0_5_pi_0_5_pi_cos) {
  std::vector<int> n = {1000};
  std::vector<double> a = {-0.5 * std::numbers::pi};
  std::vector<double> b = {0.5 * std::numbers::pi};
  double eps = 1e-4;
  std::vector<double> out(1);
  std::shared_ptr<ppc::core::TaskData> task_seq = std::make_shared<ppc::core::TaskData>();
  task_seq->inputs_count.emplace_back(n.size());
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_seq->outputs_count.emplace_back(out.size());
  poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential tmp_task_seq(task_seq, F1cos);
  ASSERT_TRUE(tmp_task_seq.ValidationImpl());
  tmp_task_seq.PreProcessingImpl();
  tmp_task_seq.RunImpl();
  tmp_task_seq.PostProcessingImpl();
  ASSERT_NEAR(2.0, out[0], eps);
}

TEST(poroshin_v_multi_integral_with_trapez_method_seq, Eulers_integral) {
  std::vector<int> n = {1000};
  std::vector<double> a = {0};
  std::vector<double> b = {0.5 * std::numbers::pi};
  double eps = 1e-4;
  std::vector<double> out(1);
  std::shared_ptr<ppc::core::TaskData> task_seq = std::make_shared<ppc::core::TaskData>();
  task_seq->inputs_count.emplace_back(n.size());
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_seq->outputs_count.emplace_back(out.size());
  poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential tmp_task_seq(task_seq, F1Euler);
  ASSERT_TRUE(tmp_task_seq.ValidationImpl());
  tmp_task_seq.PreProcessingImpl();
  tmp_task_seq.RunImpl();
  tmp_task_seq.PostProcessingImpl();
  ASSERT_NEAR(1.0, out[0], eps);
}

TEST(poroshin_v_multi_integral_with_trapez_method_seq, 05x05_area) {
  std::vector<int> n = {1000, 1000};
  std::vector<double> a = {0, 0};
  std::vector<double> b = {0.5, 0.5};
  double eps = 1e-4;
  std::vector<double> out(1);
  std::shared_ptr<ppc::core::TaskData> task_seq = std::make_shared<ppc::core::TaskData>();
  task_seq->inputs_count.emplace_back(n.size());
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_seq->outputs_count.emplace_back(out.size());
  poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential tmp_task_seq(task_seq, Area);
  ASSERT_TRUE(tmp_task_seq.ValidationImpl());
  tmp_task_seq.PreProcessingImpl();
  tmp_task_seq.RunImpl();
  tmp_task_seq.PostProcessingImpl();
  ASSERT_NEAR(0.25, out[0], eps);
}

TEST(poroshin_v_multi_integral_with_trapez_method_seq, 05x05_xy) {
  std::vector<int> n = {1000, 1000};
  std::vector<double> a = {0, 0};
  std::vector<double> b = {0.5, 0.5};
  double eps = 1e-4;
  std::vector<double> out(1);
  std::shared_ptr<ppc::core::TaskData> task_seq = std::make_shared<ppc::core::TaskData>();
  task_seq->inputs_count.emplace_back(n.size());
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_seq->outputs_count.emplace_back(out.size());
  poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential tmp_task_seq(task_seq, F2);
  ASSERT_TRUE(tmp_task_seq.ValidationImpl());
  tmp_task_seq.PreProcessingImpl();
  tmp_task_seq.RunImpl();
  tmp_task_seq.PostProcessingImpl();
  ASSERT_NEAR(0.015625, out[0], eps);
}

TEST(poroshin_v_multi_integral_with_trapez_method_seq, _0_04x0_04_tg_x_arctan_y) {
  std::vector<int> n = {1000, 1000};
  std::vector<double> a = {0, 0};
  std::vector<double> b = {0.4, 0.4};
  double eps = 1e-4;
  std::vector<double> out(1);
  std::shared_ptr<ppc::core::TaskData> task_seq = std::make_shared<ppc::core::TaskData>();
  task_seq->inputs_count.emplace_back(n.size());
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_seq->outputs_count.emplace_back(out.size());
  poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential tmp_task_seq(task_seq, F2advanced);
  ASSERT_TRUE(tmp_task_seq.ValidationImpl());
  tmp_task_seq.PreProcessingImpl();
  tmp_task_seq.RunImpl();
  tmp_task_seq.PostProcessingImpl();
  ASSERT_NEAR(0.006413250740706, out[0], eps);
}

TEST(poroshin_v_multi_integral_with_trapez_method_seq, 2x2_area) {
  std::vector<int> n = {1000, 1000};
  std::vector<double> a = {0, 0};
  std::vector<double> b = {2.0, 2.0};
  double eps = 1e-4;
  std::vector<double> out(1);
  std::shared_ptr<ppc::core::TaskData> task_seq = std::make_shared<ppc::core::TaskData>();
  task_seq->inputs_count.emplace_back(n.size());
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_seq->outputs_count.emplace_back(out.size());
  poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential tmp_task_seq(task_seq, Area);
  ASSERT_TRUE(tmp_task_seq.ValidationImpl());
  tmp_task_seq.PreProcessingImpl();
  tmp_task_seq.RunImpl();
  tmp_task_seq.PostProcessingImpl();
  ASSERT_NEAR(4.0, out[0], eps);
}

TEST(poroshin_v_multi_integral_with_trapez_method_seq, 2_3x1_4_area) {
  std::vector<int> n = {1000, 1000};
  std::vector<double> a = {2.0, 1.0};
  std::vector<double> b = {3.0, 4.0};
  double eps = 1e-4;
  std::vector<double> out(1);
  std::shared_ptr<ppc::core::TaskData> task_seq = std::make_shared<ppc::core::TaskData>();
  task_seq->inputs_count.emplace_back(n.size());
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_seq->outputs_count.emplace_back(out.size());
  poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential tmp_task_seq(task_seq, Area);
  ASSERT_TRUE(tmp_task_seq.ValidationImpl());
  tmp_task_seq.PreProcessingImpl();
  tmp_task_seq.RunImpl();
  tmp_task_seq.PostProcessingImpl();
  ASSERT_NEAR(3.0, out[0], eps);
}

TEST(poroshin_v_multi_integral_with_trapez_method_seq, _0_2xminus2_0_area) {
  std::vector<int> n = {1000, 1000};
  std::vector<double> a = {0.0, -2.0};
  std::vector<double> b = {2.0, 0.0};
  double eps = 1e-4;
  std::vector<double> out(1);
  std::shared_ptr<ppc::core::TaskData> task_seq = std::make_shared<ppc::core::TaskData>();
  task_seq->inputs_count.emplace_back(n.size());
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_seq->outputs_count.emplace_back(out.size());
  poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential tmp_task_seq(task_seq, Area);
  ASSERT_TRUE(tmp_task_seq.ValidationImpl());
  tmp_task_seq.PreProcessingImpl();
  tmp_task_seq.RunImpl();
  tmp_task_seq.PostProcessingImpl();
  ASSERT_NEAR(4.0, out[0], eps);
}

TEST(poroshin_v_multi_integral_with_trapez_method_seq, minus03_0_x_15_17_x_2_21_area) {
  std::vector<int> n = {100, 100, 100};
  std::vector<double> a = {-0.3, 1.5, 2.0};
  std::vector<double> b = {0.0, 1.7, 2.1};
  double eps = 1e-4;
  std::vector<double> out(1);
  std::shared_ptr<ppc::core::TaskData> task_seq = std::make_shared<ppc::core::TaskData>();
  task_seq->inputs_count.emplace_back(n.size());
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_seq->outputs_count.emplace_back(out.size());
  poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential tmp_task_seq(task_seq, Area);
  ASSERT_TRUE(tmp_task_seq.ValidationImpl());
  tmp_task_seq.PreProcessingImpl();
  tmp_task_seq.RunImpl();
  tmp_task_seq.PostProcessingImpl();
  ASSERT_NEAR(0.006, out[0], eps);
}

TEST(poroshin_v_multi_integral_with_trapez_method_seq, 08_1_x_15_17_x_18_2_xyz) {
  std::vector<int> n = {100, 100, 100};
  std::vector<double> a = {0.8, 1.5, 1.8};
  std::vector<double> b = {1.0, 1.7, 2.0};
  double eps = 1e-4;
  std::vector<double> out(1);
  std::shared_ptr<ppc::core::TaskData> task_seq = std::make_shared<ppc::core::TaskData>();
  task_seq->inputs_count.emplace_back(n.size());
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_seq->outputs_count.emplace_back(out.size());
  poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential tmp_task_seq(task_seq, F3);
  ASSERT_TRUE(tmp_task_seq.ValidationImpl());
  tmp_task_seq.PreProcessingImpl();
  tmp_task_seq.RunImpl();
  tmp_task_seq.PostProcessingImpl();
  ASSERT_NEAR(0.021888, out[0], eps);
}

TEST(poroshin_v_multi_integral_with_trapez_method_seq, 08_1_x_19_2_x_29_3_sinx_tgy_lnz) {
  std::vector<int> n = {100, 100, 100};
  std::vector<double> a = {0.8, 1.9, 2.9};
  std::vector<double> b = {1.0, 2.0, 3.0};
  double eps = 1e-4;
  std::vector<double> out(1);
  std::shared_ptr<ppc::core::TaskData> task_seq = std::make_shared<ppc::core::TaskData>();
  task_seq->inputs_count.emplace_back(n.size());
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_seq->outputs_count.emplace_back(out.size());
  poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential tmp_task_seq(task_seq, F3advanced);
  ASSERT_TRUE(tmp_task_seq.ValidationImpl());
  tmp_task_seq.PreProcessingImpl();
  tmp_task_seq.RunImpl();
  tmp_task_seq.PostProcessingImpl();
  ASSERT_NEAR(-0.00427191467841401, out[0], eps);
}

TEST(poroshin_v_multi_integral_with_trapez_method_seq, _0_x_05_0_y_05_0_z_05_0_w_05) {
  std::vector<int> n = {100, 100, 100, 100};
  std::vector<double> a = {0.0, 0.0, 0.0, 0.0};
  std::vector<double> b = {0.5, 0.5, 0.5, 0.5};
  double eps = 1e-6;
  std::vector<double> out(1);
  std::shared_ptr<ppc::core::TaskData> task_seq = std::make_shared<ppc::core::TaskData>();
  task_seq->inputs_count.emplace_back(n.size());
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_seq->outputs_count.emplace_back(out.size());
  poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential tmp_task_seq(task_seq, F4);
  ASSERT_TRUE(tmp_task_seq.ValidationImpl());
  tmp_task_seq.PreProcessingImpl();
  tmp_task_seq.RunImpl();
  tmp_task_seq.PostProcessingImpl();
  ASSERT_NEAR(0.000244140625, out[0], eps);
}

TEST(poroshin_v_multi_integral_with_trapez_method_seq, _0_xx_1_plus_0_yy_1_plus_0_zz_1_plus_0_ww_1) {
  std::vector<int> n = {100, 100, 100, 100};
  std::vector<double> a = {0.0, 0.0, 0.0, 0.0};
  std::vector<double> b = {1, 1, 1, 1};
  double eps = 1e-4;
  std::vector<double> out(1);
  std::shared_ptr<ppc::core::TaskData> task_seq = std::make_shared<ppc::core::TaskData>();
  task_seq->inputs_count.emplace_back(n.size());
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_seq->outputs_count.emplace_back(out.size());
  poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential tmp_task_seq(task_seq, F4advanced);
  ASSERT_TRUE(tmp_task_seq.ValidationImpl());
  tmp_task_seq.PreProcessingImpl();
  tmp_task_seq.RunImpl();
  tmp_task_seq.PostProcessingImpl();
  ASSERT_NEAR(1.3333, out[0], eps);
}