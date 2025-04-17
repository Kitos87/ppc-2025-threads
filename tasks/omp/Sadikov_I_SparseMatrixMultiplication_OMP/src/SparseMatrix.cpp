#include "omp/Sadikov_I_SparseMatrixMultiplication_OMP/include/SparseMatrix.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace sadikov_i_sparse_matrix_multiplication_task_omp {
SparseMatrix SparseMatrix::Transpose(const SparseMatrix& matrix) {
  std::vector<double> val;
  std::vector<int> rows;
  std::vector<int> elem_sum;
  auto max_size = std::max(matrix.GetRowsCount(), matrix.GetColumnsCount());
  std::vector<std::vector<double>> intermediate_values(max_size);
  std::vector<std::vector<int>> intermediate_indexes(max_size);
  int counter = 0;
  for (size_t i = 0; i < matrix.GetElementsSum().size(); ++i) {
    auto limit = i == 0 ? matrix.GetElementsSum()[0] : matrix.GetElementsSum()[i] - matrix.GetElementsSum()[i - 1];
    for (int j = 0; j < limit; ++j) {
      intermediate_values[matrix.GetRows()[counter]].emplace_back(matrix.GetValues()[counter]);
      intermediate_indexes[matrix.GetRows()[counter]].emplace_back(i);
      counter++;
    }
  }
  for (size_t i = 0; i < intermediate_values.size(); ++i) {
    for (size_t j = 0; j < intermediate_values[i].size(); ++j) {
      val.emplace_back(intermediate_values[i][j]);
      rows.emplace_back(intermediate_indexes[i][j]);
    }
    if (i > 0) {
      elem_sum.emplace_back(intermediate_values[i].size() + elem_sum[i - 1]);
    } else {
      elem_sum.emplace_back(intermediate_values[i].size());
    }
  }
  return SparseMatrix(matrix.GetColumnsCount(), matrix.GetRowsCount(), val, rows, elem_sum);
}
double SparseMatrix::CalculateSum(SparseMatrix& fmatrix, SparseMatrix& smatrix, const std::vector<int>& felements_sum,
                                  const std::vector<int>& selements_sum, int i_index, int j_index) {
  int fmatrix_elements_count = GetElementsCount(j_index, felements_sum);
  int smatrix_elements_count = GetElementsCount(i_index, selements_sum);
  int fmatrix_start_index = j_index != 0 ? felements_sum[j_index] - fmatrix_elements_count : 0;
  int smatrix_start_index = i_index != 0 ? selements_sum[i_index] - smatrix_elements_count : 0;
  double sum = 0.0;
  for (int i = 0; i < fmatrix_elements_count; i++) {
    for (int j = 0; j < smatrix_elements_count; j++) {
      if (fmatrix.GetRows()[fmatrix_start_index + i] == smatrix.GetRows()[smatrix_start_index + j]) {
        sum += fmatrix.GetValues()[i + fmatrix_start_index] * smatrix.GetValues()[j + smatrix_start_index];
      }
    }
  }
  return sum;
}

SparseMatrix SparseMatrix::operator*(SparseMatrix& smatrix) const {
  std::vector<double> values;
  std::vector<int> rows;
  std::vector<int> elements_sum(smatrix.GetColumnsCount());
  auto fmatrix = Transpose(*this);
  const auto& felements_sum = fmatrix.GetElementsSum();
  const auto& selements_sum = smatrix.GetElementsSum();
  std::vector<std::vector<std::pair<double, int>>> intermediate_values(18);
#pragma omp parallel
  {
    std::vector<std::pair<double, int>> thread_data;
#pragma omp for
    for (int i = 0; i < static_cast<int>(selements_sum.size()); ++i) {
      for (int j = 0; j < static_cast<int>(felements_sum.size()); ++j) {
        double sum = CalculateSum(fmatrix, smatrix, felements_sum, selements_sum, i, j);
        if (sum > kMEpsilon) {
          thread_data.emplace_back(sum, j);
          elements_sum[i]++;
        }
      }
    }
    intermediate_values[omp_get_thread_num()] = thread_data;
  }
  for (auto&& it : intermediate_values) {
    for (auto&& it2 : it) {
      values.emplace_back(it2.first);
      rows.emplace_back(it2.second);
    }
  }
  for (size_t i = 1; i < elements_sum.size(); ++i) {
    elements_sum[i] = elements_sum[i] + elements_sum[i - 1];
  }
  return SparseMatrix(smatrix.GetColumnsCount(), smatrix.GetColumnsCount(), values, rows, elements_sum);
}

SparseMatrix MatrixToSparse(int rows_count, int columns_count, const std::vector<double>& values) {
  std::vector<double> val;
  std::vector<int> sums(columns_count, 0);
  std::vector<int> rows;
  for (int i = 0; i < columns_count; ++i) {
    for (int j = 0; j < rows_count; ++j) {
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
  return SparseMatrix(rows_count, columns_count, val, rows, sums);
}

std::vector<double> FromSparseMatrix(const SparseMatrix& matrix) {
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

int SparseMatrix::GetElementsCount(int index, const std::vector<int>& elements_sum) {
  if (index == 0) {
    return elements_sum[index];
  }
  return elements_sum[index] - elements_sum[index - 1];
}

std::vector<double> BaseMatrixMultiplication(const std::vector<double>& fmatrix, int fmatrix_rows_count,
                                             int fmatrix_columns_count, const std::vector<double>& smatrix,
                                             int smatrix_rows_count, int smatrix_columns_count) {
  std::vector<double> answer(fmatrix_rows_count * smatrix_columns_count);
  for (int i = 0; i < fmatrix_rows_count; i++) {
    for (int j = 0; j < smatrix_columns_count; j++) {
      for (int n = 0; n < smatrix_rows_count; n++) {
        answer[j + (i * smatrix_columns_count)] +=
            fmatrix[(i * fmatrix_columns_count) + n] * smatrix[(n * smatrix_columns_count) + j];
      }
    }
  }
  return answer;
}
}  // namespace sadikov_i_sparse_matrix_multiplication_task_omp