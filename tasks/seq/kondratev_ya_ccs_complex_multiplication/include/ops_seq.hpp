#pragma once

#include <complex>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kondratev_ya_ccs_complex_multiplication_seq {

constexpr double kEpsilon = 1e-10;
constexpr double kEpsilonForZero = kEpsilon * kEpsilon;

bool IsZero(const std::complex<double>& value);
bool IsEqual(const std::complex<double>& a, const std::complex<double>& b);

struct CCSMatrix {
  std::vector<std::complex<double>> values;
  std::vector<int> row_index;
  std::vector<int> col_ptrs;
  int rows, cols;

  CCSMatrix() : rows(0), cols(0) {}
  CCSMatrix(std::pair<int, int> sizes) : rows(sizes.first), cols(sizes.second) { col_ptrs.resize(cols + 1, 0); }
  CCSMatrix operator*(const CCSMatrix& other) const;
};

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  CCSMatrix a_, b_, c_;
};

}  // namespace kondratev_ya_ccs_complex_multiplication_seq