#include "seq/tyurin_m_matmul_crs_complex/include/ops_seq.hpp"

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
MatrixCRS TransposeMatrixCRS(const MatrixCRS &crs) {
  const auto new_cols = crs.GetRows();

  MatrixCRS res;
  res.cols_count = new_cols;
  res.rowptr.resize(crs.GetCols() + 2);
  res.colind.resize(crs.colind.size(), 0);
  res.data.resize(crs.data.size(), 0);

  for (uint32_t i = 0; i < crs.data.size(); ++i) {
    ++res.rowptr[crs.colind[i] + 2];
  }
  for (uint32_t i = 2; i < res.rowptr.size(); ++i) {
    res.rowptr[i] += res.rowptr[i - 1];
  }
  for (uint32_t i = 0; i < new_cols; ++i) {
    for (uint32_t j = crs.rowptr[i]; j < crs.rowptr[i + 1]; ++j) {
      const auto new_index = res.rowptr[crs.colind[j] + 1]++;
      res.data[new_index] = crs.data[j];
      res.colind[new_index] = i;
    }
  }
  res.rowptr.pop_back();

  return res;
}
}  // namespace

bool tyurin_m_matmul_crs_complex_seq::TestTaskSequential::ValidationImpl() {
  const bool left_cols_equal_right_rows = task_data->inputs_count[1] == task_data->inputs_count[2];
  const bool there_are_rows_and_cols =
      task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0 && task_data->inputs_count[2] > 0;
  return left_cols_equal_right_rows && there_are_rows_and_cols && task_data->outputs_count[0] == 1;
}

bool tyurin_m_matmul_crs_complex_seq::TestTaskSequential::PreProcessingImpl() {
  lhs_ = *reinterpret_cast<MatrixCRS *>(task_data->inputs[0]);
  rhs_ = TransposeMatrixCRS(*reinterpret_cast<MatrixCRS *>(task_data->inputs[1]));
  res_ = {};
  res_.rowptr.resize(lhs_.GetRows() + 1);
  res_.cols_count = rhs_.GetRows();
  return true;
}

bool tyurin_m_matmul_crs_complex_seq::TestTaskSequential::RunImpl() {
  const auto rows = lhs_.GetRows();
  const auto cols = rhs_.GetRows();

  for (uint32_t i = 0; i < rows; ++i) {
    for (uint32_t j = 0; j < cols; ++j) {
      auto ii = lhs_.rowptr[i];
      auto ij = rhs_.rowptr[j];
      std::complex<double> summul = 0.0;
      while (ii < lhs_.rowptr[i + 1] && ij < rhs_.rowptr[j + 1]) {
        if (lhs_.colind[ii] < rhs_.colind[ij]) {
          ++ii;
        } else if (lhs_.colind[ii] > rhs_.colind[ij]) {
          ++ij;
        } else {
          summul += lhs_.data[ii++] * rhs_.data[ij++];
        }
      }
      if (summul != 0.0) {
        res_.data.push_back(summul);
        res_.colind.push_back(j);
      }
    }
    res_.rowptr[i + 1] = res_.data.size();
  }

  return true;
}

bool tyurin_m_matmul_crs_complex_seq::TestTaskSequential::PostProcessingImpl() {
  *reinterpret_cast<MatrixCRS *>(task_data->outputs[0]) = res_;
  return true;
}
