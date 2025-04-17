#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sorokin_a_multiplication_sparse_matrices_double_ccs_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int M_;
  int K_;
  int N_;
  std::vector<double> A_values_;
  std::vector<int> A_row_indices_;
  std::vector<int> A_col_ptr_;
  std::vector<double> B_values_;
  std::vector<int> B_row_indices_;
  std::vector<int> B_col_ptr_;
  std::vector<double> C_values_;
  std::vector<int> C_row_indices_;
  std::vector<int> C_col_ptr_;
};

void MultiplyCCS(const std::vector<double> &a_values, const std::vector<int> &a_row_indices, int m,
                 const std::vector<int> &a_col_ptr, const std::vector<double> &b_values,
                 const std::vector<int> &b_row_indices, int k, const std::vector<int> &b_col_ptr,
                 std::vector<double> &c_values, std::vector<int> &c_row_indices, int n, std::vector<int> &c_col_ptr);

}  // namespace sorokin_a_multiplication_sparse_matrices_double_ccs_seq