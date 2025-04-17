#pragma once

#include <cmath>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/Sadikov_I_SparesMatrixMultiplication/include/SparesMatrix.hpp"

namespace sadikov_i_sparse_matrix_multiplication_task_seq {

std::vector<double> GetRandomMatrix(int size);
class CCSMatrixSequential : public ppc::core::Task {
  SparesMatrix m_fMatrix_;
  SparesMatrix m_sMatrix_;
  SparesMatrix m_answerMatrix_;
  // restart tests
 public:
  explicit CCSMatrixSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace sadikov_i_sparse_matrix_multiplication_task_seq