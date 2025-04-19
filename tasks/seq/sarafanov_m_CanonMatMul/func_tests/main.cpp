#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/sarafanov_m_CanonMatMul/include/ops_seq.hpp"

TEST(sarafanov_m_canon_mat_mul_seq, test_clear_matrix) {
  constexpr size_t kCount = 0;
  constexpr double kInaccuracy = 0.001;
  std::vector<double> a_matrix;
  std::vector<double> b_matrix;
  std::vector<double> test_data;
  std::vector<double> out(kCount, 0);
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_matrix.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_matrix.data()));
  for (int i = 0; i < 4; ++i) {
    task_data_seq->inputs_count.emplace_back(kCount);
  }
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(kCount);

  sarafanov_m_canon_mat_mul_seq::CanonMatMulSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], test_data[i], kInaccuracy);
  }
}

TEST(sarafanov_m_canon_mat_mul_seq, test_1x1_matrix) {
  constexpr size_t kCount = 1;
  constexpr double kInaccuracy = 0.001;
  std::vector<double> a_matrix{18.0};
  std::vector<double> b_matrix{18.0};
  std::vector<double> test_data{324.0};
  std::vector<double> out(kCount, 0);
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_matrix.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_matrix.data()));
  for (int i = 0; i < 4; ++i) {
    task_data_seq->inputs_count.emplace_back(kCount);
  }
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(kCount);

  sarafanov_m_canon_mat_mul_seq::CanonMatMulSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], test_data[i], kInaccuracy);
  }
}

TEST(sarafanov_m_canon_mat_mul_seq, test_2x2_matrix) {
  constexpr size_t kCount = 2;
  constexpr double kInaccuracy = 0.001;
  std::vector<double> a_matrix{5.0, 6.0, 6.0, 5.0};
  std::vector<double> b_matrix{10.0, 2.0, 2.0, 10.0};
  std::vector<double> test_data{62.0, 70.0, 70.0, 62.0};
  std::vector<double> out(kCount * kCount, 0);
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_matrix.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_matrix.data()));
  for (int i = 0; i < 4; ++i) {
    task_data_seq->inputs_count.emplace_back(kCount);
  }
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(kCount * kCount);

  sarafanov_m_canon_mat_mul_seq::CanonMatMulSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], test_data[i], kInaccuracy);
  }
}

TEST(sarafanov_m_canon_mat_mul_seq, test_3x3_matrix) {
  constexpr size_t kCount = 3;
  constexpr double kInaccuracy = 0.001;
  std::vector<double> a_matrix{1.0, 4.0, 8.0, 5.0, 6.0, 2.0, 2.0, 7.0, 7.0};
  std::vector<double> b_matrix{9.0, 1.0, 10.0, 12.0, 5.0, 2.0, 9.0, 7.0, 1.0};
  std::vector<double> test_data{129.0, 77.0, 26.0, 135.0, 49.0, 64.0, 165.0, 86.0, 41.0};
  std::vector<double> out(kCount * kCount, 0);
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_matrix.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_matrix.data()));
  for (int i = 0; i < 4; ++i) {
    task_data_seq->inputs_count.emplace_back(kCount);
  }
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(kCount * kCount);

  sarafanov_m_canon_mat_mul_seq::CanonMatMulSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], test_data[i], kInaccuracy);
  }
}

TEST(sarafanov_m_canon_mat_mul_seq, test_random_5x5_matrix) {
  constexpr size_t kCount = 5;
  constexpr double kInaccuracy = 0.001;
  auto a_matrix = sarafanov_m_canon_mat_mul_seq::GenerateRandomData(static_cast<int>(kCount * kCount));
  auto single_matrix = sarafanov_m_canon_mat_mul_seq::GenerateSingleMatrix(static_cast<int>(kCount * kCount));
  std::vector<double> out(kCount * kCount, 0);
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_matrix.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(single_matrix.data()));
  for (int i = 0; i < 4; ++i) {
    task_data_seq->inputs_count.emplace_back(kCount);
  }
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(kCount * kCount);

  sarafanov_m_canon_mat_mul_seq::CanonMatMulSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], a_matrix[i], kInaccuracy);
  }
}

TEST(sarafanov_m_canon_mat_mul_seq, test_random_30x30_matrix) {
  constexpr size_t kCount = 30;
  constexpr double kInaccuracy = 0.001;
  auto a_matrix = sarafanov_m_canon_mat_mul_seq::GenerateRandomData(static_cast<int>(kCount * kCount));
  auto single_matrix = sarafanov_m_canon_mat_mul_seq::GenerateSingleMatrix(static_cast<int>(kCount * kCount));
  std::vector<double> out(kCount * kCount, 0);
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_matrix.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(single_matrix.data()));
  for (int i = 0; i < 4; ++i) {
    task_data_seq->inputs_count.emplace_back(kCount);
  }
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(kCount * kCount);

  sarafanov_m_canon_mat_mul_seq::CanonMatMulSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], a_matrix[i], kInaccuracy);
  }
}

TEST(sarafanov_m_canon_mat_mul_seq, test_2x3_matrix) {
  constexpr double kInaccuracy = 0.001;
  std::vector<double> a_matrix{1.0, 4.0, 8.0, 5.0, 6.0, 2.0};
  std::vector<double> b_matrix{9.0, 1.0, 10.0, 12.0, 5.0, 2.0};
  std::vector<double> test_data{57.0, 21.0, 18.0, 132.0, 33.0, 90.0, 78.0, 16.0, 64.0};
  std::vector<double> out(9, 0);
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_matrix.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_matrix.data()));
  task_data_seq->inputs_count.emplace_back(3);
  task_data_seq->inputs_count.emplace_back(2);
  task_data_seq->inputs_count.emplace_back(2);
  task_data_seq->inputs_count.emplace_back(3);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(9);

  sarafanov_m_canon_mat_mul_seq::CanonMatMulSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(out[i], test_data[i], kInaccuracy);
  }
}