#pragma once

#include <cmath>
#include <utility>

#include "core/task/include/task.hpp"
#include "omp/Sadikov_I_SparseMatrixMultiplication_OMP/include/SparseMatrix.hpp"

namespace sadikov_i_sparse_matrix_multiplication_task_omp {

class CCSMatrixOMP : public ppc::core::Task {
  SparseMatrix m_fMatrix_;
  SparseMatrix m_sMatrix_;
  SparseMatrix m_answerMatrix_;

 public:
  // restart tests
  explicit CCSMatrixOMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace sadikov_i_sparse_matrix_multiplication_task_omp