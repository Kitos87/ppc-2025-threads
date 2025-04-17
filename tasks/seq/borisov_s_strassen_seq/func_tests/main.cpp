#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/borisov_s_strassen_seq/include/ops_seq.hpp"

namespace {

std::vector<double> MultiplyNaiveDouble(const std::vector<double>& a, const std::vector<double>& b, int rows_a,
                                        int cols_a, int cols_b) {
  std::vector<double> c(rows_a * cols_b, 0.0);
  for (int i = 0; i < rows_a; ++i) {
    for (int j = 0; j < cols_b; ++j) {
      double sum = 0.0;
      for (int k = 0; k < cols_a; ++k) {
        sum += a[(i * cols_a) + k] * b[(k * cols_b) + j];
      }
      c[(i * cols_b) + j] = sum;
    }
  }
  return c;
}

std::vector<double> GenerateRandomMatrix(int rows, int cols, int seed, double min_val = 0.0, double max_val = 1.0) {
  std::mt19937 rng(seed);

  std::uniform_real_distribution<double> dist(min_val, max_val);
  std::vector<double> matrix(rows * cols);
  for (double& x : matrix) {
    x = dist(rng);
  }
  return matrix;
}

}  // namespace

TEST(borisov_s_strassen_seq, OneByOne) {
  std::vector<double> in_data = {1.0, 1.0, 1.0, 1.0, 7.5, 2.5};

  std::size_t output_count = 3;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data->inputs_count.push_back(in_data.size());

  auto* out_ptr = new double[output_count]();
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_ptr));
  task_data->outputs_count.push_back(output_count);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
  task.RunImpl();
  task.PostProcessingImpl();

  EXPECT_DOUBLE_EQ(out_ptr[0], 1.0);
  EXPECT_DOUBLE_EQ(out_ptr[1], 1.0);
  EXPECT_DOUBLE_EQ(out_ptr[2], 18.75);

  delete[] out_ptr;
}

TEST(borisov_s_strassen_seq, TwoByTwo) {
  std::vector<double> a = {1.0, 2.5, 3.0, 4.0};
  std::vector<double> b = {1.5, 2.0, 0.5, 3.5};
  std::vector<double> c_expected = {2.75, 10.75, 6.5, 20.0};

  std::vector<double> in_data = {2.0, 2.0, 2.0, 2.0};
  in_data.insert(in_data.end(), a.begin(), a.end());
  in_data.insert(in_data.end(), b.begin(), b.end());

  std::size_t output_count = 2 + 4;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data->inputs_count.push_back(in_data.size());

  auto* out_ptr = new double[output_count]();
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_ptr));
  task_data->outputs_count.push_back(output_count);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
  task.RunImpl();
  task.PostProcessingImpl();

  EXPECT_DOUBLE_EQ(out_ptr[0], 2.0);
  EXPECT_DOUBLE_EQ(out_ptr[1], 2.0);
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(out_ptr[2 + i], c_expected[i], 1e-9);
  }

  delete[] out_ptr;
}

TEST(borisov_s_strassen_seq, Rectangular2x3_3x4) {
  std::vector<double> a = {1.0, 2.5, 3.0, 4.0, 5.5, 6.0};
  std::vector<double> b = {0.5, 1.0, 2.0, 1.5, 2.0, 0.5, 1.0, 3.0, 4.0, 2.5, 0.5, 1.0};

  auto c_expected = MultiplyNaiveDouble(a, b, 2, 3, 4);

  std::vector<double> in_data = {2.0, 3.0, 3.0, 4.0};
  in_data.insert(in_data.end(), a.begin(), a.end());
  in_data.insert(in_data.end(), b.begin(), b.end());

  std::size_t output_count = 2 + (2 * 4);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data->inputs_count.push_back(in_data.size());

  auto* out_ptr = new double[output_count]();
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_ptr));
  task_data->outputs_count.push_back(output_count);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
  task.RunImpl();
  task.PostProcessingImpl();

  EXPECT_DOUBLE_EQ(out_ptr[0], 2.0);
  EXPECT_DOUBLE_EQ(out_ptr[1], 4.0);

  std::vector<double> c_result(2 * 4);
  for (int i = 0; i < 2 * 4; ++i) {
    c_result[i] = out_ptr[2 + i];
  }

  ASSERT_EQ(c_expected.size(), c_result.size());
  for (std::size_t i = 0; i < c_expected.size(); ++i) {
    EXPECT_NEAR(c_expected[i], c_result[i], 1e-9);
  }

  delete[] out_ptr;
}

TEST(borisov_s_strassen_seq, Square5x5_Random) {
  const int n = 5;

  std::vector<double> a = GenerateRandomMatrix(n, n, 7777, 0.0, 1.0);
  std::vector<double> b = GenerateRandomMatrix(n, n, 7777, 0.0, 1.0);

  auto c_expected = MultiplyNaiveDouble(a, b, n, n, n);

  std::vector<double> in_data = {static_cast<double>(n), static_cast<double>(n), static_cast<double>(n),
                                 static_cast<double>(n)};
  in_data.insert(in_data.end(), a.begin(), a.end());
  in_data.insert(in_data.end(), b.begin(), b.end());

  std::size_t output_count = 2 + (n * n);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data->inputs_count.push_back(in_data.size());

  auto* out_ptr = new double[output_count]();
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_ptr));
  task_data->outputs_count.push_back(output_count);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
  task.RunImpl();
  task.PostProcessingImpl();

  EXPECT_DOUBLE_EQ(out_ptr[0], static_cast<double>(n));
  EXPECT_DOUBLE_EQ(out_ptr[1], static_cast<double>(n));

  std::vector<double> c_result(n * n);
  for (int i = 0; i < n * n; ++i) {
    c_result[i] = out_ptr[2 + i];
  }

  ASSERT_EQ(c_expected.size(), c_result.size());
  for (std::size_t i = 0; i < c_expected.size(); ++i) {
    EXPECT_NEAR(c_expected[i], c_result[i], 1e-9);
  }

  delete[] out_ptr;
}

TEST(borisov_s_strassen_seq, Square20x20_Random) {
  const int n = 20;

  std::vector<double> a = GenerateRandomMatrix(n, n, 7777, 0.0, 1.0);
  std::vector<double> b = GenerateRandomMatrix(n, n, 7777, 0.0, 1.0);

  auto c_expected = MultiplyNaiveDouble(a, b, n, n, n);

  std::vector<double> in_data = {static_cast<double>(n), static_cast<double>(n), static_cast<double>(n),
                                 static_cast<double>(n)};
  in_data.insert(in_data.end(), a.begin(), a.end());
  in_data.insert(in_data.end(), b.begin(), b.end());

  std::size_t output_count = 2 + (n * n);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data->inputs_count.push_back(in_data.size());

  auto* out_ptr = new double[output_count]();
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_ptr));
  task_data->outputs_count.push_back(output_count);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
  task.RunImpl();
  task.PostProcessingImpl();

  EXPECT_DOUBLE_EQ(out_ptr[0], static_cast<double>(n));
  EXPECT_DOUBLE_EQ(out_ptr[1], static_cast<double>(n));

  std::vector<double> c_result(n * n);
  for (int i = 0; i < n * n; ++i) {
    c_result[i] = out_ptr[2 + i];
  }

  ASSERT_EQ(c_expected.size(), c_result.size());
  for (std::size_t i = 0; i < c_expected.size(); ++i) {
    EXPECT_NEAR(c_expected[i], c_result[i], 1e-9);
  }

  delete[] out_ptr;
}

TEST(borisov_s_strassen_seq, Square32x32_Random) {
  const int n = 32;

  std::vector<double> a = GenerateRandomMatrix(n, n, 7777, 0.0, 1.0);
  std::vector<double> b = GenerateRandomMatrix(n, n, 7777, 0.0, 1.0);

  auto c_expected = MultiplyNaiveDouble(a, b, n, n, n);

  std::vector<double> in_data = {static_cast<double>(n), static_cast<double>(n), static_cast<double>(n),
                                 static_cast<double>(n)};
  in_data.insert(in_data.end(), a.begin(), a.end());
  in_data.insert(in_data.end(), b.begin(), b.end());

  std::size_t output_count = 2 + (n * n);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data->inputs_count.push_back(in_data.size());

  auto* out_ptr = new double[output_count]();
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_ptr));
  task_data->outputs_count.push_back(output_count);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
  task.RunImpl();
  task.PostProcessingImpl();

  EXPECT_DOUBLE_EQ(out_ptr[0], static_cast<double>(n));
  EXPECT_DOUBLE_EQ(out_ptr[1], static_cast<double>(n));

  std::vector<double> c_result(n * n);
  for (int i = 0; i < n * n; ++i) {
    c_result[i] = out_ptr[2 + i];
  }

  ASSERT_EQ(c_expected.size(), c_result.size());
  for (std::size_t i = 0; i < c_expected.size(); ++i) {
    EXPECT_NEAR(c_expected[i], c_result[i], 1e-9);
  }

  delete[] out_ptr;
}

TEST(borisov_s_strassen_seq, Square128x128_Random) {
  const int n = 128;

  std::vector<double> a = GenerateRandomMatrix(n, n, 7777, 0.0, 1.0);
  std::vector<double> b = GenerateRandomMatrix(n, n, 7777, 0.0, 1.0);

  auto c_expected = MultiplyNaiveDouble(a, b, n, n, n);

  std::vector<double> in_data = {static_cast<double>(n), static_cast<double>(n), static_cast<double>(n),
                                 static_cast<double>(n)};
  in_data.insert(in_data.end(), a.begin(), a.end());
  in_data.insert(in_data.end(), b.begin(), b.end());

  std::size_t output_count = 2 + (n * n);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data->inputs_count.push_back(in_data.size());

  auto* out_ptr = new double[output_count]();
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_ptr));
  task_data->outputs_count.push_back(output_count);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
  task.RunImpl();
  task.PostProcessingImpl();

  EXPECT_DOUBLE_EQ(out_ptr[0], static_cast<double>(n));
  EXPECT_DOUBLE_EQ(out_ptr[1], static_cast<double>(n));

  std::vector<double> c_result(n * n);
  for (int i = 0; i < n * n; ++i) {
    c_result[i] = out_ptr[2 + i];
  }

  ASSERT_EQ(c_expected.size(), c_result.size());
  for (std::size_t i = 0; i < c_expected.size(); ++i) {
    EXPECT_NEAR(c_expected[i], c_result[i], 1e-9);
  }

  delete[] out_ptr;
}

TEST(borisov_s_strassen_seq, Square128x128_IdentityMatrix) {
  const int n = 128;

  std::vector<double> a = GenerateRandomMatrix(n, n, 7777, 0.0, 1.0);

  std::vector<double> e(n * n, 0.0);
  for (int i = 0; i < n; ++i) {
    e[(i * n) + i] = 1.0;
  }

  std::vector<double> in_data = {static_cast<double>(n), static_cast<double>(n), static_cast<double>(n),
                                 static_cast<double>(n)};
  in_data.insert(in_data.end(), a.begin(), a.end());
  in_data.insert(in_data.end(), e.begin(), e.end());

  std::size_t output_count = 2 + (n * n);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data->inputs_count.push_back(in_data.size());

  auto* out_ptr = new double[output_count]();
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_ptr));
  task_data->outputs_count.push_back(output_count);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
  task.RunImpl();
  task.PostProcessingImpl();

  EXPECT_DOUBLE_EQ(out_ptr[0], static_cast<double>(n));
  EXPECT_DOUBLE_EQ(out_ptr[1], static_cast<double>(n));

  std::vector<double> c_result(n * n);
  for (int i = 0; i < n * n; ++i) {
    c_result[i] = out_ptr[2 + i];
  }

  ASSERT_EQ(a.size(), c_result.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_NEAR(a[i], c_result[i], 1e-9);
  }

  delete[] out_ptr;
}

TEST(borisov_s_strassen_seq, Square129x129_Random) {
  const int n = 129;

  std::vector<double> a = GenerateRandomMatrix(n, n, 7777, 0.0, 1.0);
  std::vector<double> b = GenerateRandomMatrix(n, n, 7777, 0.0, 1.0);

  auto c_expected = MultiplyNaiveDouble(a, b, n, n, n);

  std::vector<double> in_data = {static_cast<double>(n), static_cast<double>(n), static_cast<double>(n),
                                 static_cast<double>(n)};
  in_data.insert(in_data.end(), a.begin(), a.end());
  in_data.insert(in_data.end(), b.begin(), b.end());

  std::size_t output_count = 2 + (n * n);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data->inputs_count.push_back(in_data.size());

  auto* out_ptr = new double[output_count]();
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_ptr));
  task_data->outputs_count.push_back(output_count);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
  task.RunImpl();
  task.PostProcessingImpl();

  EXPECT_DOUBLE_EQ(out_ptr[0], static_cast<double>(n));
  EXPECT_DOUBLE_EQ(out_ptr[1], static_cast<double>(n));

  std::vector<double> c_result(n * n);
  for (int i = 0; i < n * n; ++i) {
    c_result[i] = out_ptr[2 + i];
  }

  ASSERT_EQ(c_expected.size(), c_result.size());
  for (std::size_t i = 0; i < c_expected.size(); ++i) {
    EXPECT_NEAR(c_expected[i], c_result[i], 1e-9);
  }

  delete[] out_ptr;
}

TEST(borisov_s_strassen_seq, Square240x240_Random) {
  const int n = 240;

  std::vector<double> a = GenerateRandomMatrix(n, n, 7777, 0.0, 1.0);
  std::vector<double> b = GenerateRandomMatrix(n, n, 7777, 0.0, 1.0);

  auto c_expected = MultiplyNaiveDouble(a, b, n, n, n);

  std::vector<double> in_data = {static_cast<double>(n), static_cast<double>(n), static_cast<double>(n),
                                 static_cast<double>(n)};
  in_data.insert(in_data.end(), a.begin(), a.end());
  in_data.insert(in_data.end(), b.begin(), b.end());

  std::size_t output_count = 2 + (n * n);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data->inputs_count.push_back(in_data.size());

  auto* out_ptr = new double[output_count]();
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_ptr));
  task_data->outputs_count.push_back(output_count);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
  task.RunImpl();
  task.PostProcessingImpl();

  EXPECT_DOUBLE_EQ(out_ptr[0], static_cast<double>(n));
  EXPECT_DOUBLE_EQ(out_ptr[1], static_cast<double>(n));

  std::vector<double> c_result(n * n);
  for (int i = 0; i < n * n; ++i) {
    c_result[i] = out_ptr[2 + i];
  }

  ASSERT_EQ(c_expected.size(), c_result.size());
  for (std::size_t i = 0; i < c_expected.size(); ++i) {
    EXPECT_NEAR(c_expected[i], c_result[i], 1e-9);
  }

  delete[] out_ptr;
}

TEST(borisov_s_strassen_seq, ValidCase) {
  std::vector<double> input_data = {2.0, 3.0, 3.0, 2.0};
  input_data.insert(input_data.end(), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  input_data.insert(input_data.end(), {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(input_data.size());

  task_data->outputs.push_back(nullptr);
  task_data->outputs_count.push_back(0);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();

  EXPECT_TRUE(task.ValidationImpl());
}

TEST(borisov_s_strassen_seq, MismatchCase) {
  std::vector<double> input_data = {
      2.0,
      2.0,
      3.0,
      3.0,
  };
  input_data.insert(input_data.end(), {1.0, 2.0, 3.0, 4.0});
  input_data.insert(input_data.end(), {5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0});

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(input_data.size());

  task_data->outputs.push_back(nullptr);
  task_data->outputs_count.push_back(0);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();
  EXPECT_FALSE(task.ValidationImpl());
}

TEST(borisov_s_strassen_seq, NotEnoughDataCase) {
  std::vector<double> input_data = {2.0, 2.0, 2.0, 2.0};
  input_data.insert(input_data.end(), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->inputs_count.push_back(input_data.size());

  task_data->outputs.push_back(nullptr);
  task_data->outputs_count.push_back(0);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();

  EXPECT_FALSE(task.ValidationImpl());
}

TEST(borisov_s_strassen_seq, Rectangular16x17_Random) {
  const int rows_a = 16;
  const int cols_a = 17;
  const int cols_b = 18;

  std::vector<double> a = GenerateRandomMatrix(rows_a, cols_a, 7777);
  std::vector<double> b = GenerateRandomMatrix(cols_a, cols_b, 7777);

  auto c_expected = MultiplyNaiveDouble(a, b, rows_a, cols_a, cols_b);

  std::vector<double> in_data = {static_cast<double>(rows_a), static_cast<double>(cols_a), static_cast<double>(cols_a),
                                 static_cast<double>(cols_b)};
  in_data.insert(in_data.end(), a.begin(), a.end());
  in_data.insert(in_data.end(), b.begin(), b.end());

  std::size_t output_count = 2 + (rows_a * cols_b);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data->inputs_count.push_back(in_data.size());

  auto* out_ptr = new double[output_count]();
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_ptr));
  task_data->outputs_count.push_back(output_count);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
  task.RunImpl();
  task.PostProcessingImpl();

  EXPECT_DOUBLE_EQ(out_ptr[0], static_cast<double>(rows_a));
  EXPECT_DOUBLE_EQ(out_ptr[1], static_cast<double>(cols_b));

  std::vector<double> c_result(rows_a * cols_b);
  for (int i = 0; i < rows_a * cols_b; ++i) {
    c_result[i] = out_ptr[2 + i];
  }

  ASSERT_EQ(c_expected.size(), c_result.size());
  for (std::size_t i = 0; i < c_expected.size(); ++i) {
    EXPECT_NEAR(c_expected[i], c_result[i], 1e-9);
  }

  delete[] out_ptr;
}

TEST(borisov_s_strassen_seq, Rectangular19x23_Random) {
  const int rows_a = 19;
  const int cols_a = 23;
  const int cols_b = 21;

  std::vector<double> a = GenerateRandomMatrix(rows_a, cols_a, 7777);
  std::vector<double> b = GenerateRandomMatrix(cols_a, cols_b, 7777);

  auto c_expected = MultiplyNaiveDouble(a, b, rows_a, cols_a, cols_b);

  std::vector<double> in_data = {static_cast<double>(rows_a), static_cast<double>(cols_a), static_cast<double>(cols_a),
                                 static_cast<double>(cols_b)};
  in_data.insert(in_data.end(), a.begin(), a.end());
  in_data.insert(in_data.end(), b.begin(), b.end());

  std::size_t output_count = 2 + (rows_a * cols_b);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data->inputs_count.push_back(in_data.size());

  auto* out_ptr = new double[output_count]();
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_ptr));
  task_data->outputs_count.push_back(output_count);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
  task.RunImpl();
  task.PostProcessingImpl();

  EXPECT_DOUBLE_EQ(out_ptr[0], static_cast<double>(rows_a));
  EXPECT_DOUBLE_EQ(out_ptr[1], static_cast<double>(cols_b));

  std::vector<double> c_result(rows_a * cols_b);
  for (int i = 0; i < rows_a * cols_b; ++i) {
    c_result[i] = out_ptr[2 + i];
  }

  ASSERT_EQ(c_expected.size(), c_result.size());
  for (std::size_t i = 0; i < c_expected.size(); ++i) {
    EXPECT_NEAR(c_expected[i], c_result[i], 1e-9);
  }

  delete[] out_ptr;
}

TEST(borisov_s_strassen_seq, Rectangular32x64_Random) {
  const int rows_a = 32;
  const int cols_a = 64;
  const int cols_b = 32;

  std::vector<double> a = GenerateRandomMatrix(rows_a, cols_a, 7777);
  std::vector<double> b = GenerateRandomMatrix(cols_a, cols_b, 7777);

  auto c_expected = MultiplyNaiveDouble(a, b, rows_a, cols_a, cols_b);

  std::vector<double> in_data = {static_cast<double>(rows_a), static_cast<double>(cols_a), static_cast<double>(cols_a),
                                 static_cast<double>(cols_b)};
  in_data.insert(in_data.end(), a.begin(), a.end());
  in_data.insert(in_data.end(), b.begin(), b.end());

  std::size_t output_count = 2 + (rows_a * cols_b);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_data.data()));
  task_data->inputs_count.push_back(in_data.size());

  auto* out_ptr = new double[output_count]();
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_ptr));
  task_data->outputs_count.push_back(output_count);

  borisov_s_strassen_seq::SequentialStrassenSeq task(task_data);

  task.PreProcessingImpl();
  EXPECT_TRUE(task.ValidationImpl());
  task.RunImpl();
  task.PostProcessingImpl();

  EXPECT_DOUBLE_EQ(out_ptr[0], static_cast<double>(rows_a));
  EXPECT_DOUBLE_EQ(out_ptr[1], static_cast<double>(cols_b));

  std::vector<double> c_result(rows_a * cols_b);
  for (int i = 0; i < rows_a * cols_b; ++i) {
    c_result[i] = out_ptr[2 + i];
  }

  ASSERT_EQ(c_expected.size(), c_result.size());
  for (std::size_t i = 0; i < c_expected.size(); ++i) {
    EXPECT_NEAR(c_expected[i], c_result[i], 1e-9);
  }

  delete[] out_ptr;
}
