#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/sarafanov_m_CanonMatMul_omp/include/CanonMatrix.hpp"

namespace sarafanov_m_canon_mat_mul_omp {

class CanonMatMulOMP : public ppc::core::Task {
  CanonMatrix a_matrix_;
  CanonMatrix b_matrix_;
  CanonMatrix c_matrix_;
  static constexpr double kInaccuracy = 0.001;

 public:
  explicit CanonMatMulOMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  bool CheckSquareSize(int number);
  static std::vector<double> ConvertToSquareMatrix(int need_size, MatrixType type, const std::vector<double>& matrx);
};
}  // namespace sarafanov_m_canon_mat_mul_omp