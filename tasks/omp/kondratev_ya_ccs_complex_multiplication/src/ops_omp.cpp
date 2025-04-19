#include "omp/kondratev_ya_ccs_complex_multiplication/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <utility>
#include <vector>

bool kondratev_ya_ccs_complex_multiplication_omp::IsZero(const std::complex<double> &value) {
  return std::norm(value) < kEpsilonForZero;
}

bool kondratev_ya_ccs_complex_multiplication_omp::IsEqual(const std::complex<double> &a,
                                                          const std::complex<double> &b) {
  return std::norm(a - b) <= kEpsilonForZero;
}

bool kondratev_ya_ccs_complex_multiplication_omp::TestTaskOMP::PreProcessingImpl() {
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

bool kondratev_ya_ccs_complex_multiplication_omp::TestTaskOMP::ValidationImpl() {
  return task_data->inputs_count[0] == 2 && task_data->outputs_count[0] == 1 && task_data->inputs[0] != nullptr &&
         task_data->inputs[1] != nullptr && task_data->outputs[0] != nullptr;
}

bool kondratev_ya_ccs_complex_multiplication_omp::TestTaskOMP::RunImpl() {
  c_ = a_ * b_;
  return true;
}

bool kondratev_ya_ccs_complex_multiplication_omp::TestTaskOMP::PostProcessingImpl() {
  *reinterpret_cast<CCSMatrix *>(task_data->outputs[0]) = c_;
  return true;
}

kondratev_ya_ccs_complex_multiplication_omp::CCSMatrix
kondratev_ya_ccs_complex_multiplication_omp::CCSMatrix::operator*(const CCSMatrix &other) const {
  CCSMatrix result({rows, other.cols});
  result.values.reserve(std::min(rows * other.cols, static_cast<int>(values.size() * other.values.size())));
  result.row_index.reserve(result.values.capacity());
  result.col_ptrs.resize(other.cols + 1, 0);

  std::vector<std::vector<std::pair<int, std::complex<double>>>> temp_cols(other.cols);

#pragma omp parallel for
  for (int result_col = 0; result_col < other.cols; result_col++) {
    std::vector<std::complex<double>> local_temp_col(rows, std::complex<double>(0.0, 0.0));

    for (int k = other.col_ptrs[result_col]; k < other.col_ptrs[result_col + 1]; k++) {
      int row_other = other.row_index[k];
      std::complex<double> val_other = other.values[k];

      for (int i = col_ptrs[row_other]; i < col_ptrs[row_other + 1]; i++) {
        int row_this = row_index[i];
        local_temp_col[row_this] += values[i] * val_other;
      }
    }

    for (int i = 0; i < rows; i++) {
      if (!IsZero(local_temp_col[i])) {
        temp_cols[result_col].emplace_back(i, local_temp_col[i]);
      }
    }
  }

  int nonzero_elements_processed = 0;
  for (int col = 0; col < other.cols; col++) {
    result.col_ptrs[col] = nonzero_elements_processed;

    for (const auto &[row, value] : temp_cols[col]) {
      result.row_index.emplace_back(row);
      result.values.emplace_back(value);
      nonzero_elements_processed++;
    }
  }
  result.col_ptrs[other.cols] = nonzero_elements_processed;

  return result;
}