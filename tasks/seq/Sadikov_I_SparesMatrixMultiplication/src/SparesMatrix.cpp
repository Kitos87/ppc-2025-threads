#include "seq/Sadikov_I_SparesMatrixMultiplication/include/SparesMatrix.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace sadikov_i_sparse_matrix_multiplication_task_seq {
SparesMatrix SparesMatrix::Transpose(const SparesMatrix& matrix) {
  std::vector<double> val;
  std::vector<int> rows;
  std::vector<int> elem_sum;
  auto max_size = std::max(matrix.GetRowsCount(), matrix.GetColumnsCount());
  std::vector<std::vector<double>> intermediate_values(max_size);
  std::vector<std::vector<int>> intermediate_indexes(max_size);
  int counter = 0;
  for (int i = 0; i < static_cast<int>(matrix.GetElementsSum().size()); ++i) {
    auto limit = i == 0 ? matrix.GetElementsSum()[0] : matrix.GetElementsSum()[i] - matrix.GetElementsSum()[i - 1];
    for (int j = 0; j < limit; ++j) {
      intermediate_values[matrix.GetRows()[counter]].emplace_back(matrix.GetValues()[counter]);
      intermediate_indexes[matrix.GetRows()[counter]].emplace_back(i);
      counter++;
    }
  }
  for (auto i = 0; i < static_cast<int>(intermediate_values.size()); ++i) {
    for (auto j = 0; j < static_cast<int>(intermediate_values[i].size()); ++j) {
      val.emplace_back(intermediate_values[i][j]);
      rows.emplace_back(intermediate_indexes[i][j]);
    }
    if (i > 0) {
      elem_sum.emplace_back(intermediate_values[i].size() + elem_sum[i - 1]);
    } else {
      elem_sum.emplace_back(intermediate_values[i].size());
    }
  }
  return SparesMatrix(matrix.GetColumnsCount(), matrix.GetRowsCount(), val, rows, elem_sum);
}

SparesMatrix SparesMatrix::operator*(const SparesMatrix& smatrix) const {
  std::vector<double> values;
  std::vector<int> rows;
  std::vector<int> elements_sum(smatrix.GetColumnsCount());
  auto fmatrix = Transpose(*this);
  const auto& felements_sum = fmatrix.GetElementsSum();
  const auto& selements_sum = smatrix.GetElementsSum();
  for (auto i = 0; i < static_cast<int>(selements_sum.size()); ++i) {
    for (auto j = 0; j < static_cast<int>(felements_sum.size()); ++j) {
      auto sum = 0.0;
      auto fmatrix_elements_count = GetElementsCount(j, felements_sum);
      auto smatrix_elements_count = GetElementsCount(i, selements_sum);
      auto fmatrix_start_index = j != 0 ? felements_sum[j] - fmatrix_elements_count : 0;
      auto smatrix_start_index = i != 0 ? selements_sum[i] - smatrix_elements_count : 0;

      for (auto n = 0; n < fmatrix_elements_count; n++) {
        for (auto n2 = 0; n2 < smatrix_elements_count; n2++) {
          if (fmatrix.GetRows()[fmatrix_start_index + n] == smatrix.GetRows()[smatrix_start_index + n2]) {
            sum += fmatrix.GetValues()[n + fmatrix_start_index] * smatrix.GetValues()[n2 + smatrix_start_index];
          }
        }
      }
      if (sum > kMEpsilon) {
        values.emplace_back(sum);
        rows.emplace_back(j);
        elements_sum[i]++;
      }
    }
  }
  for (auto i = 1; i < static_cast<int>(elements_sum.size()); ++i) {
    elements_sum[i] = elements_sum[i] + elements_sum[i - 1];
  }
  return SparesMatrix(smatrix.GetColumnsCount(), smatrix.GetColumnsCount(), values, rows, elements_sum);
}

SparesMatrix MatrixToSpares(int rows_count, int columns_count, const std::vector<double>& values) {
  std::vector<double> val;
  std::vector<int> sums(columns_count, 0);
  std::vector<int> rows;
  for (auto i = 0; i < columns_count; ++i) {
    for (auto j = 0; j < rows_count; ++j) {
      if (values[i + (columns_count * j)] != 0) {
        val.emplace_back(values[i + (columns_count * j)]);
        rows.emplace_back(j);
        sums[i] += 1;
      }
    }
    if (i != columns_count - 1) {
      sums[i + 1] = sums[i];
    }
  }
  return SparesMatrix(rows_count, columns_count, val, rows, sums);
}

std::vector<double> FromSparesMatrix(const SparesMatrix& matrix) {
  std::vector<double> simple_matrix(matrix.GetRowsCount() * matrix.GetColumnsCount(), 0.0);
  int counter = 0;
  for (size_t i = 0; i < matrix.GetElementsSum().size(); ++i) {
    auto limit = i == 0 ? matrix.GetElementsSum()[0] : matrix.GetElementsSum()[i] - matrix.GetElementsSum()[i - 1];
    for (int j = 0; j < limit; ++j) {
      simple_matrix[i + (matrix.GetColumnsCount() * matrix.GetRows()[counter])] = matrix.GetValues()[counter];
      counter++;
    }
  }
  return simple_matrix;
}

int SparesMatrix::GetElementsCount(int index, const std::vector<int>& elements_sum) {
  if (index == 0) {
    return elements_sum[index];
  }
  return elements_sum[index] - elements_sum[index - 1];
}

std::vector<double> BaseMatrixMultiplication(const std::vector<double>& fmatrix, int fmatrix_rows_count,
                                             int fmatrix_columns_count, const std::vector<double>& smatrix,
                                             int smatrix_rows_count, int smatrix_columns_count) {
  std::vector<double> answer(fmatrix_rows_count * smatrix_columns_count);
  for (auto i = 0; i < fmatrix_rows_count; i++) {
    for (auto j = 0; j < smatrix_columns_count; j++) {
      for (auto n = 0; n < smatrix_rows_count; n++) {
        answer[j + (i * smatrix_columns_count)] +=
            fmatrix[(i * fmatrix_columns_count) + n] * smatrix[(n * smatrix_columns_count) + j];
      }
    }
  }
  return answer;
}
}  // namespace sadikov_i_sparse_matrix_multiplication_task_seq