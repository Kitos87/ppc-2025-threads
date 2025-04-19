#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace korotin_e_crs_multiplication_seq {

class CrsMultiplicationSequential : public ppc::core::Task {
 public:
  explicit CrsMultiplicationSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> A_val_, B_val_, output_val_;
  std::vector<unsigned int> A_col_, A_rI_, B_col_, B_rI_, output_col_, output_rI_;
  unsigned int A_N_, A_Nz_, B_N_, B_Nz_;
};

std::vector<double> GetRandomMatrix(unsigned int m, unsigned int n);

void MakeCRS(std::vector<unsigned int> &r_i, std::vector<unsigned int> &col, std::vector<double> &val,
             const std::vector<double> &src, unsigned int m, unsigned int n);

void MatrixMultiplication(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &c,
                          unsigned int m, unsigned int n, unsigned int p);

}  // namespace korotin_e_crs_multiplication_seq
