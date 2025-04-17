#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/solovev_a_ccs_mmult_sparse/include/ccs_mmult_sparse.hpp"

namespace {
std::complex<double> GenerateRandomComplex(double min, double max) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(min, max);
  return {dis(gen), dis(gen)};
}

bool AreComplexNumbersApproxEqual(const std::complex<double>& c1, const std::complex<double>& c2,
                                  double tolerance = 1e-6) {
  return std::abs(c1.real() - c2.real()) < tolerance && std::abs(c1.imag() - c2.imag()) < tolerance;
}
}  // namespace

TEST(solovev_a_ccs_mmult_sparse, test_I) {
  solovev_a_matrix::MatrixInCcsSparse m1(1, 1, 1);
  solovev_a_matrix::MatrixInCcsSparse m2(1, 1, 1);
  solovev_a_matrix::MatrixInCcsSparse m3;

  m1.col_p = {0, 1};
  m1.row = {0};
  m1.val = {std::complex<double>(0.0, 1.0)};

  m2.col_p = {0, 1};
  m2.row = {0};
  m2.val = {std::complex<double>(0.0, -1.0)};

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&m1));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&m2));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&m3));

  solovev_a_matrix::SeqMatMultCcs test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  std::complex<double> correct_reply(1.0, 0.0);
  ASSERT_NEAR(std::abs(m3.val[0] - correct_reply), 0.0, 1e-6);
}

TEST(solovev_a_ccs_mmult_sparse, test_II) {
  std::complex<double> vvector(1.0, 1.0);
  const int rows_m1 = 50;
  const int cols_m1 = 1;
  const int rows_m2 = 1;
  const int cols_m2 = 50;
  const int rows_m3 = rows_m1;
  const int cols_m3 = cols_m2;
  const int nnz_m3 = rows_m3 * cols_m3;

  solovev_a_matrix::MatrixInCcsSparse m1(rows_m1, cols_m1, rows_m1);
  solovev_a_matrix::MatrixInCcsSparse m2(rows_m2, cols_m2, cols_m2);
  solovev_a_matrix::MatrixInCcsSparse m3(rows_m3, cols_m3, nnz_m3);

  m1.col_p = {0, rows_m1};

  for (int i = 0; i <= cols_m2; i++) {
    m2.col_p.push_back(i);
  }
  for (int i = 0; i < rows_m1; i++) {
    m1.row.push_back(i);
    m1.val.emplace_back(vvector);
  }
  for (int i = 0; i < cols_m2; i++) {
    m2.row.push_back(0);
    m2.val.emplace_back(vvector);
  }

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&m1));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&m2));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&m3));

  solovev_a_matrix::SeqMatMultCcs test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  std::complex<double> correct_reply(0.0, 0.0);
  for (int i = 0; i < nnz_m3; i++) {
    ASSERT_EQ(m3.val[i], correct_reply);
  }
}

TEST(solovev_a_ccs_mmult_sparse, test_III) {
  std::complex<double> vvector(2.0, 1.0);
  solovev_a_matrix::MatrixInCcsSparse m1(50, 50);
  solovev_a_matrix::MatrixInCcsSparse m2(50, 1);
  solovev_a_matrix::MatrixInCcsSparse m3(50, 1);

  int l = 1;
  int m = 0;

  for (int i = 0; i <= 50; i++) {
    m1.col_p.push_back(m);
    m += l;
    l++;
  }

  l = 1;
  m = 0;
  for (int i = 0; i < m1.col_p[50]; i++) {
    m1.val.emplace_back(vvector);
    if (m >= l) {
      m = 0;
      l++;
    }
    m1.row.push_back(m);
    m++;
  }

  m2.col_p = {0, 50};
  for (int i = 0; i < 50; i++) {
    m2.val.emplace_back(vvector);
    m2.row.push_back(i);
  }

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&m1));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&m2));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&m3));

  solovev_a_matrix::SeqMatMultCcs test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  l = 50;
  for (size_t i = 0; i < m3.val.size(); i++) {
    ASSERT_EQ(m3.val[i], std::complex<double>(3.0 * l, 4.0 * l));
    l--;
  }
}

TEST(solovev_a_ccs_mmult_sparse, test_IV) {
  std::complex<double> vvector_one(2.0, 1.0);
  std::complex<double> vvector_two(3.0, 4.0);
  solovev_a_matrix::MatrixInCcsSparse m1(5, 5);
  solovev_a_matrix::MatrixInCcsSparse m2(5, 5);
  solovev_a_matrix::MatrixInCcsSparse m3(5, 5);

  m1.col_p = {0, 0, 1, 1, 1, 1};
  m2.col_p = {0, 0, 1, 1, 1, 1};

  m1.val = {vvector_one};
  m1.row = {1};
  m2.val = {vvector_two};
  m2.row = {1};

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&m1));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&m2));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&m3));

  solovev_a_matrix::SeqMatMultCcs test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  std::complex<double> correct_reply(2.0, 11.0);

  ASSERT_EQ(m3.val[0], correct_reply);
}

TEST(solovev_a_ccs_mmult_sparse, test_V) {
  std::complex<double> vvector(2.0, 1.0);
  solovev_a_matrix::MatrixInCcsSparse m1(50, 50);
  solovev_a_matrix::MatrixInCcsSparse m2(50, 50);
  solovev_a_matrix::MatrixInCcsSparse m3(50, 50);

  for (int i = 0; i <= 50; i++) {
    m1.col_p.push_back(i);
    m2.col_p.push_back(i);
  }

  for (int i = 0; i < 50; i++) {
    m1.row.push_back(i);
    m1.val.emplace_back(vvector);
    m2.row.push_back(i);
    m2.val.emplace_back(vvector);
  };

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&m1));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&m2));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&m3));

  solovev_a_matrix::SeqMatMultCcs test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  std::complex<double> correct_reply(3.0, 4.0);
  for (size_t i = 0; i < m3.val.size(); i++) {
    ASSERT_EQ(m3.val[i], correct_reply);
  }
}

TEST(solovev_a_ccs_mmult_sparse, test_V_random) {
  int rows = 50;
  int cols = 50;

  solovev_a_matrix::MatrixInCcsSparse m1(rows, cols);
  solovev_a_matrix::MatrixInCcsSparse m2(rows, cols);
  solovev_a_matrix::MatrixInCcsSparse m3(rows, cols);

  for (int i = 0; i <= cols; i++) {
    m1.col_p.push_back(i);
    m2.col_p.push_back(i);
  }

  for (int i = 0; i < rows; i++) {
    m1.row.push_back(i);
    m1.val.push_back(GenerateRandomComplex(-10.0, 10.0));
    m2.row.push_back(i);
    m2.val.push_back(GenerateRandomComplex(-10.0, 10.0));
  }

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&m1));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&m2));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&m3));

  solovev_a_matrix::SeqMatMultCcs test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();

  for (size_t i = 0; i < m3.val.size(); i++) {
    bool approx_equal = AreComplexNumbersApproxEqual(m3.val[i], m1.val[i] * m2.val[i]);
    ASSERT_TRUE(approx_equal);
  }
}

TEST(solovev_a_ccs_mmult_sparse, test_identity_multiplication) {
  const int size = 50;
  solovev_a_matrix::MatrixInCcsSparse i1(size, size, size);
  solovev_a_matrix::MatrixInCcsSparse a(size, size, size * size);
  solovev_a_matrix::MatrixInCcsSparse result(size, size, 0);

  i1.col_p.resize(size + 1);
  for (int i = 0; i < size; i++) {
    i1.col_p[i] = i;
    i1.row.push_back(i);
    i1.val.emplace_back(1.0, 0.0);
  }
  i1.col_p[size] = size;

  a.col_p.resize(size + 1);
  int nz_count = 0;
  for (int j = 0; j < size; j++) {
    a.col_p[j] = nz_count;
    for (int i = 0; i < size; i++) {
      a.row.push_back(i);
      a.val.emplace_back(static_cast<double>(i + j), static_cast<double>(i - j));
      nz_count++;
    }
  }
  a.col_p[size] = nz_count;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&i1));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));

  solovev_a_matrix::SeqMatMultCcs multiplication_task(task_data);
  ASSERT_EQ(multiplication_task.ValidationImpl(), true);
  multiplication_task.PreProcessingImpl();
  multiplication_task.RunImpl();
  multiplication_task.PostProcessingImpl();

  for (size_t i = 0; i < result.val.size(); i++) {
    ASSERT_EQ(result.val[i], a.val[i]);
  }
}
