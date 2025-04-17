
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/odintsov_m_multmatrix_cannon/include/ops_seq.hpp"

TEST(odintsov_m_mulmatrix_cannon_seq, test_matrix_4) {
  // Create data
  std::vector<double> matrix_a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<double> matrix_b = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<double> out(16, 0);
  std::vector<double> matrix_c = {90, 100, 110, 120, 202, 228, 254, 280, 314, 356, 398, 440, 426, 484, 542, 600};

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_seq->inputs_count.emplace_back(matrix_a.size());
  task_data_seq->inputs_count.emplace_back(matrix_a.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));

  // Create Task
  odintsov_m_mulmatrix_cannon_seq::MulMatrixCannonSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();

  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, matrix_c);
}
TEST(odintsov_m_mulmatrix_cannon_seq, test_matrix_100) {
  // Create data
  std::vector<double> matrix_a(100, 1);
  std::vector<double> matrix_b(100, 1);
  std::vector<double> out(100, 0);
  std::vector<double> matrix_c(100, 10);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_seq->inputs_count.emplace_back(matrix_a.size());
  task_data_seq->inputs_count.emplace_back(matrix_a.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));

  // Create Task
  odintsov_m_mulmatrix_cannon_seq::MulMatrixCannonSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();

  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, matrix_c);
}

TEST(odintsov_m_mulmatrix_cannon_seq, test_matrix_900) {
  // Create data
  std::vector<double> matrix_a(900, 1);
  std::vector<double> matrix_b(900, 1);
  std::vector<double> out(900, 0);
  std::vector<double> matrix_c(900, 30);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_seq->inputs_count.emplace_back(matrix_a.size());
  task_data_seq->inputs_count.emplace_back(matrix_a.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));

  // Create Task
  odintsov_m_mulmatrix_cannon_seq::MulMatrixCannonSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();

  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, matrix_c);
}
TEST(odintsov_m_mulmatrix_cannon_seq, test_sz_block_1) {
  // Create data
  std::vector<double> matrix_a(9, 1);
  std::vector<double> matrix_b(9, 1);
  std::vector<double> out(9, 0);
  std::vector<double> matrix_c(9, 3);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_seq->inputs_count.emplace_back(matrix_a.size());
  task_data_seq->inputs_count.emplace_back(matrix_a.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));

  // Create Task
  odintsov_m_mulmatrix_cannon_seq::MulMatrixCannonSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();

  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, matrix_c);
}
TEST(odintsov_m_mulmatrix_cannon_seq, test_validation) {
  // Create data
  std::vector<double> matrix_a(12, 1);
  std::vector<double> matrix_b(12, 1);
  std::vector<double> out(12, 0);
  std::vector<double> matrix_c(12, 3);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_seq->inputs_count.emplace_back(matrix_a.size());
  task_data_seq->inputs_count.emplace_back(matrix_a.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));

  // Create Task
  odintsov_m_mulmatrix_cannon_seq::MulMatrixCannonSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}

TEST(odintsov_m_mulmatrix_cannon_seq, test_sz_prime) {
  // Create data
  std::vector<double> matrix_a(13 * 13, 1);
  std::vector<double> matrix_b(13 * 13, 1);
  std::vector<double> out(13 * 13, 0);
  std::vector<double> matrix_c(13 * 13, 13);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_seq->inputs_count.emplace_back(matrix_a.size());
  task_data_seq->inputs_count.emplace_back(matrix_a.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));

  // Create Task
  odintsov_m_mulmatrix_cannon_seq::MulMatrixCannonSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();

  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, matrix_c);
}

TEST(odintsov_m_mulmatrix_cannon_seq, test_sz_13) {
  // Create data
  std::vector<double> matrix_a(13, 1);
  std::vector<double> matrix_b(13, 1);
  std::vector<double> out(13, 0);
  std::vector<double> matrix_c(13, 13);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
  task_data_seq->inputs_count.emplace_back(matrix_a.size());
  task_data_seq->inputs_count.emplace_back(matrix_a.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));

  // Create Task
  odintsov_m_mulmatrix_cannon_seq::MulMatrixCannonSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}