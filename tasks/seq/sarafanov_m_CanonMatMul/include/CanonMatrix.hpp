#pragma once

#include <cstddef>
#include <vector>

enum class MatrixType : char { kRowMatrix, kColumnMatrix };

namespace sarafanov_m_canon_mat_mul_seq {
class CanonMatrix {
  std::vector<double> matrix_;
  size_t size_ = 0;

  void CalculateSize(size_t s);
  [[nodiscard]] size_t GetRowIndex(size_t index, size_t row_number) const;
  [[nodiscard]] size_t GetColumnIndex(size_t index, size_t column_index, size_t offset) const;

 public:
  CanonMatrix() = default;
  CanonMatrix(const std::vector<double>& initial_vector);
  void SetBaseMatrix(const std::vector<double>& initial_vector);
  void Transpose();
  void StairShift();
  void PreRoutine(MatrixType type);
  [[nodiscard]] const std::vector<double>& GetMatrix() const;
  [[nodiscard]] size_t GetSize() const;
  CanonMatrix MultiplicateMatrix(const CanonMatrix& canon_matrix, size_t offset);
  CanonMatrix operator+(const CanonMatrix& canon_matrix);
  void ClearMatrix();
};
}  // namespace sarafanov_m_canon_mat_mul_seq