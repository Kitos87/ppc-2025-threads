#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/sarafanov_m_CanonMatMul/include/CanonMatrix.hpp"

namespace sarafanov_m_canon_mat_mul_seq {

class CanonMatMulSequential : public ppc::core::Task {
  CanonMatrix a_matrix_;
  CanonMatrix b_matrix_;
  CanonMatrix c_matrix_;
  static constexpr double kInaccuracy = 0.001;

 public:
  explicit CanonMatMulSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  bool CheckSquareSize(int number);
  static std::vector<double> ConvertToSquareMatrix(int need_size, MatrixType type, const std::vector<double>& matrx);
};
std::vector<double> GenerateRandomData(int size);
std::vector<double> GenerateSingleMatrix(int size);
}  // namespace sarafanov_m_canon_mat_mul_seq