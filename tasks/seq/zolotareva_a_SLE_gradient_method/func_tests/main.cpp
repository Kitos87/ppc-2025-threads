#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/zolotareva_a_SLE_gradient_method/include/ops_seq.hpp"

void zolotareva_a_sle_gradient_method_seq::GenerateSle(std::vector<double> &a, std::vector<double> &b, int n) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-100.0, 100.0);

  for (int i = 0; i < n; ++i) {
    b[i] = dist(gen);
    for (int j = i; j < n; ++j) {
      double value = dist(gen);
      a[(i * n) + j] = value;
      a[(j * n) + i] = value;
    }
  }

  for (int i = 0; i < n; ++i) {
    a[(i * n) + i] += n * 100.0;
  }
}

namespace {
void Form(int n) {
  std::vector<double> a(n * n);
  std::vector<double> b(n);
  std::vector<double> x(n);
  zolotareva_a_sle_gradient_method_seq::GenerateSle(a, b, n);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs_count.push_back(n * n);
  task_data_seq->inputs_count.push_back(n);
  task_data_seq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
  task_data_seq->outputs_count.push_back(x.size());

  zolotareva_a_sle_gradient_method_seq::TestTaskSequential task(task_data_seq);
  ASSERT_EQ(task.ValidationImpl(), true);
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();

  for (int i = 0; i < n; ++i) {
    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
      sum += a[(i * n) + j] * x[j];
    }
    EXPECT_NEAR(sum, b[i], 1e-4);
  }
}
}  // namespace

TEST(zolotareva_a_sle_gradient_method_seq, invalid_input_sizes) {
  int n = 2;
  std::vector<double> a = {2, -1, -1, 2};
  std::vector<double> b = {1, 3, 4};  // Неправильный размер b
  std::vector<double> x(n);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs_count.push_back(n * n);
  task_data_seq->inputs_count.push_back(b.size());
  task_data_seq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
  task_data_seq->outputs_count.push_back(n);

  zolotareva_a_sle_gradient_method_seq::TestTaskSequential task(task_data_seq);
  ASSERT_FALSE(task.ValidationImpl());
}

TEST(zolotareva_a_sle_gradient_method_seq, non_symmetric_matrix) {
  int n = 2;
  std::vector<double> a = {2, -1, 0, 2};  // a[0][1] != a[1][0]
  std::vector<double> b = {1, 3};
  std::vector<double> x(n);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs_count.push_back(n * n);
  task_data_seq->inputs_count.push_back(n);
  task_data_seq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
  task_data_seq->outputs_count.push_back(n);

  zolotareva_a_sle_gradient_method_seq::TestTaskSequential task(task_data_seq);
  ASSERT_EQ(task.ValidationImpl(), false);
}

TEST(zolotareva_a_sle_gradient_method_seq, not_positive_definite_matrix) {
  int n = 2;
  std::vector<double> a = {0, 0, 0, 0};
  std::vector<double> b = {0, 0};
  std::vector<double> x(n);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs_count.push_back(n * n);
  task_data_seq->inputs_count.push_back(n);
  task_data_seq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
  task_data_seq->outputs_count.push_back(n);

  zolotareva_a_sle_gradient_method_seq::TestTaskSequential task(task_data_seq);
  ASSERT_EQ(task.ValidationImpl(), false);
}

TEST(zolotareva_a_sle_gradient_method_seq, negative_definite_matrix) {
  int n = 2;
  std::vector<double> a = {-1, 0, 0, -2};
  std::vector<double> b = {1, 1};
  std::vector<double> x(n);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs_count.push_back(n * n);
  task_data_seq->inputs_count.push_back(n);
  task_data_seq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
  task_data_seq->outputs_count.push_back(n);

  zolotareva_a_sle_gradient_method_seq::TestTaskSequential task(task_data_seq);
  ASSERT_EQ(task.ValidationImpl(), false);
}

TEST(zolotareva_a_sle_gradient_method_seq, zero_dimension) {
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> x;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs_count.push_back(0);
  task_data_seq->inputs_count.push_back(0);
  task_data_seq->outputs_count.push_back(0);

  zolotareva_a_sle_gradient_method_seq::TestTaskSequential task(task_data_seq);
  ASSERT_EQ(task.ValidationImpl(), false);
}

TEST(zolotareva_a_sle_gradient_method_seq, singular_matrix) {
  int n = 2;
  std::vector<double> a = {1, 1, 1, 1};  // Сингулярная матрица
  std::vector<double> b = {2, 2};
  std::vector<double> x(n);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs_count.push_back(n * n);
  task_data_seq->inputs_count.push_back(n);
  task_data_seq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
  task_data_seq->outputs_count.push_back(n);

  zolotareva_a_sle_gradient_method_seq::TestTaskSequential task(task_data_seq);
  ASSERT_EQ(task.ValidationImpl(), false);
}

TEST(zolotareva_a_sle_gradient_method_seq, zero_vector_solution) {
  int n = 2;
  std::vector<double> a = {1, 0, 0, 1};
  std::vector<double> b = {0, 0};
  std::vector<double> x(n);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs_count.push_back(n * n);
  task_data_seq->inputs_count.push_back(n);
  task_data_seq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
  task_data_seq->outputs_count.push_back(n);

  zolotareva_a_sle_gradient_method_seq::TestTaskSequential task(task_data_seq);
  ASSERT_EQ(task.ValidationImpl(), true);
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();

  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(x[i], 0.0, 1e-2);  // Ожидаем нулевой вектор решения
  }
}

TEST(zolotareva_a_sle_gradient_method_seq, n_equals_one) {
  int n = 1;
  std::vector<double> a = {2};
  std::vector<double> b = {4};
  std::vector<double> x(n);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs_count.push_back(n * n);
  task_data_seq->inputs_count.push_back(n);
  task_data_seq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
  task_data_seq->outputs_count.push_back(n);

  zolotareva_a_sle_gradient_method_seq::TestTaskSequential task(task_data_seq);
  ASSERT_EQ(task.ValidationImpl(), true);
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();

  EXPECT_NEAR(x[0], 2.0, 1e-1);  // Ожидаемое решение x = 2
}

TEST(zolotareva_a_sle_gradient_method_seq, test_correct_answer1) {
  int n = 3;
  std::vector<double> a = {4, -1, 2, -1, 6, -2, 2, -2, 5};
  std::vector<double> b = {-1, 9, -10};
  std::vector<double> x;
  x.resize(n);
  std::vector<double> ref_x = {1, 1, -2};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_seq->inputs_count.push_back(n * n);
  task_data_seq->inputs_count.push_back(n);
  task_data_seq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
  task_data_seq->outputs_count.push_back(x.size());

  zolotareva_a_sle_gradient_method_seq::TestTaskSequential task(task_data_seq);
  ASSERT_EQ(task.ValidationImpl(), true);
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();
  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(x[i], ref_x[i], 1e-12);
  }
}
TEST(zolotareva_a_sle_gradient_method_seq, Test_Image_random_n_3) { Form(3); };
TEST(zolotareva_a_sle_gradient_method_seq, Test_Image_random_n_5) { Form(5); };
TEST(zolotareva_a_sle_gradient_method_seq, Test_Image_random_n_7) { Form(7); };
TEST(zolotareva_a_sle_gradient_method_seq, Test_Image_random_n_20) { Form(591); };
