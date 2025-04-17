#pragma once

#include <complex>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

using Complex = std::complex<double>;

namespace kolodkin_g_multiplication_matrix_seq {

struct SparseMatrixCRS {
  std::vector<Complex> values;
  std::vector<int> colIndices;
  std::vector<int> rowPtr;
  int numRows;
  int numCols;
  SparseMatrixCRS() : values({}), colIndices({}), rowPtr({}), numRows(0), numCols(0) {};
  SparseMatrixCRS(int rows, int cols) : numRows(rows), numCols(cols) { rowPtr.resize(rows + 1, 0); }

  void AddValue(int row, Complex value, int col);
  SparseMatrixCRS(const SparseMatrixCRS& other) = default;
  SparseMatrixCRS& operator=(const SparseMatrixCRS& other) = default;
  static void PrintSparseMatrix(const SparseMatrixCRS& matrix);
};
std::vector<Complex> ParseMatrixIntoVec(const SparseMatrixCRS& mat);
SparseMatrixCRS ParseVectorIntoMatrix(std::vector<Complex>& vec);
bool CheckMatrixesEquality(const SparseMatrixCRS& a, const SparseMatrixCRS& b);
bool AreEqualElems(const Complex& a, const Complex& b, double epsilon);
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<Complex> input_, output_;
  SparseMatrixCRS A_, B_;
};

}  // namespace kolodkin_g_multiplication_matrix_seq