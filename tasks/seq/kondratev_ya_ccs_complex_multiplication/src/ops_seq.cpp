#include "seq/kondratev_ya_ccs_complex_multiplication/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

bool kondratev_ya_ccs_complex_multiplication_seq::IsZero(const std::complex<double> &value) {
  return std::norm(value) < kEpsilonForZero;
}

bool kondratev_ya_ccs_complex_multiplication_seq::IsEqual(const std::complex<double> &a,
                                                          const std::complex<double> &b) {
  return std::norm(a - b) <= kEpsilonForZero;
}

bool kondratev_ya_ccs_complex_multiplication_seq::TestTaskSequential::PreProcessingImpl() {
  a_ = *reinterpret_cast<CCSMatrix *>(task_data->inputs[0]);
  b_ = *reinterpret_cast<CCSMatrix *>(task_data->inputs[1]);

  if (a_.rows == 0 || a_.cols == 0 || b_.rows == 0 || b_.cols == 0) {
    return false;
  }

  if (a_.cols != b_.rows) {
    return false;
  }

  return true;
}

bool kondratev_ya_ccs_complex_multiplication_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == 2 && task_data->outputs_count[0] == 1 && task_data->inputs[0] != nullptr &&
         task_data->inputs[1] != nullptr && task_data->outputs[0] != nullptr;
}

bool kondratev_ya_ccs_complex_multiplication_seq::TestTaskSequential::RunImpl() {
  c_ = a_ * b_;
  return true;
}

bool kondratev_ya_ccs_complex_multiplication_seq::TestTaskSequential::PostProcessingImpl() {
  *reinterpret_cast<CCSMatrix *>(task_data->outputs[0]) = c_;
  return true;
}

kondratev_ya_ccs_complex_multiplication_seq::CCSMatrix
kondratev_ya_ccs_complex_multiplication_seq::CCSMatrix::operator*(const CCSMatrix &other) const {
  CCSMatrix result({rows, other.cols});
  result.values.reserve(std::min(rows * other.cols, static_cast<int>(values.size() * other.values.size())));
  result.row_index.reserve(result.values.capacity());

  std::vector<std::complex<double>> temp_col(rows, std::complex<double>(0.0, 0.0));

  for (int result_col = 0; result_col < other.cols; result_col++) {
    for (int k = other.col_ptrs[result_col]; k < other.col_ptrs[result_col + 1]; k++) {
      int row_other = other.row_index[k];
      std::complex<double> val_other = other.values[k];

      for (int i = col_ptrs[row_other]; i < col_ptrs[row_other + 1]; i++) {
        int row_this = row_index[i];
        temp_col[row_this] += values[i] * val_other;
      }
    }

    result.col_ptrs[result_col] = static_cast<int>(result.values.size());
    for (int i = 0; i < rows; i++) {
      if (!IsZero(temp_col[i])) {
        result.values.emplace_back(temp_col[i]);
        result.row_index.emplace_back(i);

        temp_col[i] = std::complex<double>(0.0, 0.0);
      }
    }
  }

  result.col_ptrs[other.cols] = static_cast<int>(result.values.size());

  return result;
}