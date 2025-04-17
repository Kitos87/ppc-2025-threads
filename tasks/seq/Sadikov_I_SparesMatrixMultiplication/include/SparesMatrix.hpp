#pragma once

#include <vector>

namespace sadikov_i_sparse_matrix_multiplication_task_seq {
class SparesMatrix {
  constexpr static double kMEpsilon = 0.000001;
  int m_rowsCount_ = 0;
  int m_columnsCount_ = 0;
  std::vector<double> m_values_;
  std::vector<int> m_rows_;
  std::vector<int> m_elementsSum_;
  static SparesMatrix Transpose(const SparesMatrix& matrix);
  static int GetElementsCount(int index, const std::vector<int>& elements_sum);

 public:
  SparesMatrix() = default;
  explicit SparesMatrix(int rows_count, int columns_count, const std::vector<double>& values,
                        const std::vector<int>& rows, const std::vector<int>& element_sum) noexcept
      : m_rowsCount_(rows_count),
        m_columnsCount_(columns_count),
        m_values_(values),
        m_rows_(rows),
        m_elementsSum_(element_sum) {};
  [[nodiscard]] const std::vector<double>& GetValues() const noexcept { return m_values_; }
  [[nodiscard]] const std::vector<int>& GetRows() const noexcept { return m_rows_; }
  [[nodiscard]] const std::vector<int>& GetElementsSum() const noexcept { return m_elementsSum_; }
  [[nodiscard]] int GetColumnsCount() const noexcept { return m_columnsCount_; }
  [[nodiscard]] int GetRowsCount() const noexcept { return m_rowsCount_; }
  SparesMatrix operator*(const SparesMatrix& smatrix) const noexcept(false);
};

SparesMatrix MatrixToSpares(int rows_count, int columns_count, const std::vector<double>& values);

std::vector<double> FromSparesMatrix(const SparesMatrix& matrix);

std::vector<double> BaseMatrixMultiplication(const std::vector<double>& fmatrix, int fmatrix_rows_count,
                                             int fmatrix_columns_count, const std::vector<double>& smatrix,
                                             int smatrix_rows_count, int smatrix_columns_count);
}  // namespace sadikov_i_sparse_matrix_multiplication_task_seq