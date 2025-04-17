#pragma once

#include <complex>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

struct Matrix {
  uint32_t rows;
  uint32_t cols;
  std::vector<std::complex<double>> data;

  std::complex<double>& Get(uint32_t row, uint32_t col) { return data[(row * cols) + col]; }

  bool operator==(const Matrix& other) const noexcept {
    return rows == other.rows && cols == other.cols && data == other.data;
  }
};

inline Matrix MultiplyMat(Matrix& lhs, Matrix& rhs) {
  Matrix res{.rows = lhs.rows, .cols = rhs.cols, .data = std::vector<std::complex<double>>(lhs.rows * rhs.cols)};
  for (uint32_t i = 0; i < lhs.rows; i++) {
    for (uint32_t j = 0; j < rhs.cols; j++) {
      res.Get(i, j) = 0;
      for (uint32_t k = 0; k < rhs.rows; k++) {
        res.Get(i, j) += lhs.Get(i, k) * rhs.Get(k, j);
      }
    }
  }
  return res;
}

struct MatrixCRS {
  std::vector<std::complex<double>> data;

  uint32_t cols_count;
  std::vector<uint32_t> rowptr;
  std::vector<uint32_t> colind;

  //

  [[nodiscard]] uint32_t GetRows() const { return rowptr.size() - 1; }
  [[nodiscard]] uint32_t GetCols() const { return cols_count; }

  bool operator==(const MatrixCRS& other) const noexcept {
    return cols_count == other.cols_count && rowptr == other.rowptr && colind == other.colind && data == other.data;
  }
};

inline MatrixCRS RegularToCRS(const Matrix& matrix) {
  MatrixCRS result;
  result.rowptr.resize(matrix.rows + 1);
  result.cols_count = matrix.cols;

  uint32_t i = 0;
  for (uint32_t row = 0; row < matrix.rows; ++row) {
    uint32_t nz = 0;
    for (uint32_t col = 0; col < matrix.cols; ++col) {
      if (const auto& element = matrix.data[i++]; element != 0.0) {
        ++nz;
        result.colind.push_back(col);
        result.data.push_back(element);
      }
    }
    result.rowptr[row + 1] = result.rowptr[row] + nz;
  }

  return result;
}

inline Matrix CRSToRegular(const MatrixCRS& crs) {
  Matrix matrix{.rows = crs.GetRows(),
                .cols = crs.GetCols(),
                .data = std::vector<std::complex<double>>(crs.GetRows() * crs.GetCols())};
  for (uint32_t row = 0; row < matrix.rows; ++row) {
    for (uint32_t i = crs.rowptr[row]; i < crs.rowptr[row + 1]; ++i) {
      matrix.Get(row, crs.colind[i]) = crs.data[i];
    }
  }
  return matrix;
}

namespace tyurin_m_matmul_crs_complex_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  MatrixCRS lhs_;
  MatrixCRS rhs_;
  MatrixCRS res_;
};

}  // namespace tyurin_m_matmul_crs_complex_seq