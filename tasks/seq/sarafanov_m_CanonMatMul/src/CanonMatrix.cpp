#include "seq/sarafanov_m_CanonMatMul/include/CanonMatrix.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "utility"

namespace sarafanov_m_canon_mat_mul_seq {
CanonMatrix::CanonMatrix(const std::vector<double>& initial_vector) : matrix_(initial_vector) {
  CalculateSize(initial_vector.size());
}
void CanonMatrix::CalculateSize(size_t s) { size_ = static_cast<int>(std::sqrt(s)); }

void CanonMatrix::PreRoutine(MatrixType type) {
  if (type == MatrixType::kColumnMatrix) {
    Transpose();
  }
  StairShift();
}

void CanonMatrix::SetBaseMatrix(const std::vector<double>& initial_vector) {
  if (matrix_.empty()) {
    CalculateSize(initial_vector.size());
    matrix_ = initial_vector;
  }
}

size_t CanonMatrix::GetRowIndex(size_t index, size_t row_number) const {
  if (index < size_ * row_number) {
    return index;
  }
  auto shift = index - (size_ * row_number);
  return (row_number * size_) - size_ + shift;
}

size_t CanonMatrix::GetColumnIndex(size_t index, size_t column_index, size_t offset) const {
  if (index + offset < size_ * column_index) {
    return index + offset;
  }
  auto shift = index + offset - (size_ * column_index);
  return (size_ * (column_index - 1)) + shift;
}

void CanonMatrix::StairShift() {
  std::vector<double> new_matrix(matrix_.size());
  std::copy(matrix_.begin(), matrix_.begin() + static_cast<int>(size_), new_matrix.begin());
  int s_size = static_cast<int>(size_);
  for (int i = 1; i < s_size; ++i) {
    std::copy(matrix_.begin() + s_size * i + i, matrix_.begin() + s_size * (i + 1), new_matrix.begin() + s_size * i);
    for (int j = s_size * i; j < s_size * i + i; ++j) {
      new_matrix[j + s_size - i] = matrix_[j];
    }
  }
  matrix_ = std::move(new_matrix);
}

const std::vector<double>& CanonMatrix::GetMatrix() const { return matrix_; }

CanonMatrix CanonMatrix::MultiplicateMatrix(const CanonMatrix& canon_matrix, size_t offset) {
  std::vector<double> c_matrix(size_ * size_);
  const auto& b_matrix = canon_matrix.GetMatrix();
  for (size_t i = 0; i < size_; ++i) {
    for (size_t j = 0; j < size_; ++j) {
      c_matrix[(i * size_) + j] = matrix_[GetRowIndex((i * size_) + j + offset, i + 1)] *
                                  b_matrix[GetColumnIndex((j * size_) + i, j + 1, offset)];
    }
  }
  return {c_matrix};
}

size_t CanonMatrix::GetSize() const { return size_; }

CanonMatrix CanonMatrix::operator+(const CanonMatrix& canon_matrix) {
  std::vector<double> c_matrix(canon_matrix.GetSize() * canon_matrix.GetSize());
  if (this->GetMatrix().empty()) {
    SetBaseMatrix(c_matrix);
  }
  const auto& b_matrix = canon_matrix.GetMatrix();
  for (size_t i = 0; i < size_; ++i) {
    for (size_t j = 0; j < size_; ++j) {
      c_matrix[(i * size_) + j] = matrix_[(i * size_) + j] + b_matrix[(j * size_) + i];
    }
  }
  return {c_matrix};
}

void CanonMatrix::Transpose() {
  std::vector<double> new_matrix(size_ * size_);
  for (size_t i = 0; i < size_; ++i) {
    for (size_t j = 0; j < size_; ++j) {
      new_matrix[(i * size_) + j] = matrix_[(j * size_) + i];
    }
  }
  matrix_ = std::move(new_matrix);
}

void CanonMatrix::ClearMatrix() {
  matrix_.clear();
  size_ = 0;
}
}  // namespace sarafanov_m_canon_mat_mul_seq