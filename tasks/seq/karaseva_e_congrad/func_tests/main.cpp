#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/karaseva_e_congrad/include/ops_seq.hpp"

namespace {

// Function to generate a random symmetric positive-definite matrix of size matrix_size x matrix_size.
// The matrix is computed as A = R^T * R.
std::vector<double> GenerateRandomSPDMatrix(size_t matrix_size, unsigned int seed = 42) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(0.1, 1.0);
  std::vector<double> r_matrix(matrix_size * matrix_size);
  for (size_t i = 0; i < matrix_size * matrix_size; ++i) {
    r_matrix[i] = dist(gen);
  }
  std::vector<double> a_matrix(matrix_size * matrix_size, 0.0);
  // Compute a_matrix = R^T * R
  for (size_t i = 0; i < matrix_size; ++i) {
    for (size_t j = 0; j < matrix_size; ++j) {
      for (size_t k = 0; k < matrix_size; ++k) {
        a_matrix[(i * matrix_size) + j] += (r_matrix[(k * matrix_size) + i] * r_matrix[(k * matrix_size) + j]);
      }
    }
  }
  // Add diagonal dominance
  for (size_t i = 0; i < matrix_size; ++i) {
    a_matrix[(i * matrix_size) + i] += static_cast<double>(matrix_size);
  }
  return a_matrix;
}

// Helper function to multiply a_matrix (size matrix_size x matrix_size) by vector x (length matrix_size)
std::vector<double> MultiplyMatrixVector(const std::vector<double>& a_matrix, const std::vector<double>& x,
                                         size_t matrix_size) {
  std::vector<double> result(matrix_size, 0.0);
  for (size_t i = 0; i < matrix_size; ++i) {
    for (size_t j = 0; j < matrix_size; ++j) {
      result[i] += (a_matrix[(i * matrix_size) + j] * x[j]);
    }
  }
  return result;
}

}  // namespace

TEST(karaseva_e_congrad_seq, test_identity_50) {
  constexpr size_t kN = 50;

  // Create an identity matrix a_matrix of size kN x kN
  std::vector<double> a_matrix(kN * kN, 0.0);
  for (size_t i = 0; i < kN; ++i) {
    a_matrix[(i * kN) + i] = 1.0;
  }

  // Create a vector b with elements 1.0, 2.0, ..., kN
  std::vector<double> b(kN);
  for (size_t i = 0; i < kN; ++i) {
    b[i] = static_cast<double>(i + 1);
  }

  // Vector for the solution x, initially filled with zeros
  std::vector<double> x(kN, 0.0);

  // Set up task data structure
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(a_matrix.data()));
  task_data_seq->inputs_count.push_back(kN * kN);
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.push_back(kN);
  task_data_seq->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
  task_data_seq->outputs_count.push_back(kN);

  // Create task
  karaseva_e_congrad_seq::TestTaskSequential test_task(task_data_seq);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  // Check that the computed solution x matches vector b with an accuracy of 1e-9
  for (size_t i = 0; i < kN; ++i) {
    EXPECT_NEAR(x[i], b[i], 1e-9);
  }
}

TEST(karaseva_e_congrad_seq, test_random_spd_small) {
  constexpr size_t kN = 20;  // system size

  // Generate a random SPD matrix a_matrix with a fixed seed
  auto a_matrix = GenerateRandomSPDMatrix(kN, 42);

  // Generate a random true solution vector x_true
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(0.1, 1.0);
  std::vector<double> x_true(kN);
  for (size_t i = 0; i < kN; ++i) {
    x_true[i] = dist(gen);
  }

  // Compute the right-hand side b = a_matrix * x_true
  auto b = MultiplyMatrixVector(a_matrix, x_true, kN);

  // Vector for the computed solution x, initially zeros
  std::vector<double> x(kN, 0.0);

  // Set up task data structure
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(a_matrix.data()));
  task_data_seq->inputs_count.push_back(kN * kN);
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.push_back(kN);
  task_data_seq->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
  task_data_seq->outputs_count.push_back(kN);

  // Create task
  karaseva_e_congrad_seq::TestTaskSequential test_task(task_data_seq);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  // Check that the computed solution x is close to the true solution x_true with an accuracy of 1e-6
  for (size_t i = 0; i < kN; ++i) {
    EXPECT_NEAR(x[i], x_true[i], 1e-6);
  }
}

TEST(karaseva_e_congrad_seq, test_small_system_size_1) {
  constexpr size_t kN = 1;

  std::vector<double> a_matrix = {5.0};
  std::vector<double> b = {10.0};
  std::vector<double> x(kN, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(a_matrix.data()));
  task_data_seq->inputs_count.push_back(kN * kN);
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.push_back(kN);
  task_data_seq->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
  task_data_seq->outputs_count.push_back(kN);

  karaseva_e_congrad_seq::TestTaskSequential test_task(task_data_seq);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  EXPECT_NEAR(x[0], 2.0, 1e-10);
}

TEST(karaseva_e_congrad_seq, test_validation_invalid_matrix) {
  constexpr size_t kRows = 2;
  constexpr size_t kCols = 3;
  std::vector<double> a_matrix(kRows * kCols, 1.0);
  std::vector<double> b(kRows, 1.0);
  std::vector<double> x(kRows, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(a_matrix.data()));
  task_data_seq->inputs_count.push_back(kRows * kCols);
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.push_back(kRows);
  task_data_seq->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
  task_data_seq->outputs_count.push_back(kRows);

  karaseva_e_congrad_seq::TestTaskSequential test_task(task_data_seq);
  ASSERT_FALSE(test_task.Validation());
}

TEST(karaseva_e_congrad_seq, test_validation_invalid_output) {
  constexpr size_t kN = 2;
  std::vector<double> a_matrix(kN * kN, 1.0);
  std::vector<double> b(kN, 1.0);
  std::vector<double> x(3, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(a_matrix.data()));
  task_data_seq->inputs_count.push_back(kN * kN);
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.push_back(kN);
  task_data_seq->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
  task_data_seq->outputs_count.push_back(3);

  karaseva_e_congrad_seq::TestTaskSequential test_task(task_data_seq);
  ASSERT_FALSE(test_task.Validation());
}

TEST(karaseva_e_congrad_seq, test_diagonal_matrix_100) {
  constexpr size_t kN = 100;
  std::vector<double> a_matrix(kN * kN, 0.0);
  for (size_t i = 0; i < kN; ++i) {
    a_matrix[(i * kN) + i] = static_cast<double>(i + 1);
  }
  std::vector<double> b(kN);
  for (size_t i = 0; i < kN; ++i) {
    b[i] = static_cast<double>(i + 1);
  }
  std::vector<double> x(kN, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(a_matrix.data()));
  task_data_seq->inputs_count.push_back(kN * kN);
  task_data_seq->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
  task_data_seq->inputs_count.push_back(kN);
  task_data_seq->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
  task_data_seq->outputs_count.push_back(kN);

  karaseva_e_congrad_seq::TestTaskSequential test_task(task_data_seq);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  for (size_t i = 0; i < kN; ++i) {
    EXPECT_NEAR(x[i], 1.0, 1e-9);
  }
}