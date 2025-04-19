#include "seq/nasedkin_e_strassen_algorithm/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

bool nasedkin_e_strassen_algorithm_seq::StrassenSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr_a = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* in_ptr_b = reinterpret_cast<double*>(task_data->inputs[1]);

  matrix_size_ = static_cast<int>(std::sqrt(input_size));
  input_matrix_a_.resize(matrix_size_ * matrix_size_);
  input_matrix_b_.resize(matrix_size_ * matrix_size_);

  std::ranges::copy(in_ptr_a, in_ptr_a + input_size, input_matrix_a_.begin());
  std::ranges::copy(in_ptr_b, in_ptr_b + input_size, input_matrix_b_.begin());

  if ((matrix_size_ & (matrix_size_ - 1)) != 0) {
    original_size_ = matrix_size_;
    input_matrix_a_ = PadMatrixToPowerOfTwo(input_matrix_a_, matrix_size_);
    input_matrix_b_ = PadMatrixToPowerOfTwo(input_matrix_b_, matrix_size_);
    matrix_size_ = static_cast<int>(std::sqrt(input_matrix_a_.size()));
  } else {
    original_size_ = matrix_size_;
  }

  output_matrix_.resize(matrix_size_ * matrix_size_, 0.0);
  return true;
}

bool nasedkin_e_strassen_algorithm_seq::StrassenSequential::ValidationImpl() {
  unsigned int input_size_a = task_data->inputs_count[0];
  unsigned int input_size_b = task_data->inputs_count[1];
  unsigned int output_size = task_data->outputs_count[0];

  int size_a = static_cast<int>(std::sqrt(input_size_a));
  int size_b = static_cast<int>(std::sqrt(input_size_b));
  int size_output = static_cast<int>(std::sqrt(output_size));

  return (size_a == size_b) && (size_a == size_output);
}

bool nasedkin_e_strassen_algorithm_seq::StrassenSequential::RunImpl() {
  output_matrix_ = StrassenMultiply(input_matrix_a_, input_matrix_b_, matrix_size_);
  return true;
}

bool nasedkin_e_strassen_algorithm_seq::StrassenSequential::PostProcessingImpl() {
  if (original_size_ != matrix_size_) {
    output_matrix_ = TrimMatrixToOriginalSize(output_matrix_, original_size_, matrix_size_);
  }

  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  std::ranges::copy(output_matrix_, out_ptr);
  return true;
}

std::vector<double> nasedkin_e_strassen_algorithm_seq::StrassenSequential::AddMatrices(const std::vector<double>& a,
                                                                                       const std::vector<double>& b,
                                                                                       int size) {
  std::vector<double> result(size * size);
  std::ranges::transform(a, b, result.begin(), std::plus<>());
  return result;
}

std::vector<double> nasedkin_e_strassen_algorithm_seq::StrassenSequential::SubtractMatrices(
    const std::vector<double>& a, const std::vector<double>& b, int size) {
  std::vector<double> result(size * size);
  std::ranges::transform(a, b, result.begin(), std::minus<>());
  return result;
}

std::vector<double> nasedkin_e_strassen_algorithm_seq::StandardMultiply(const std::vector<double>& a,
                                                                        const std::vector<double>& b, int size) {
  std::vector<double> result(size * size, 0.0);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      for (int k = 0; k < size; ++k) {
        result[(i * size) + j] += a[(i * size) + k] * b[(k * size) + j];
      }
    }
  }
  return result;
}

std::vector<double> nasedkin_e_strassen_algorithm_seq::StrassenSequential::PadMatrixToPowerOfTwo(
    const std::vector<double>& matrix, int original_size) {
  int new_size = 1;
  while (new_size < original_size) {
    new_size *= 2;
  }

  std::vector<double> padded_matrix(new_size * new_size, 0);
  for (int i = 0; i < original_size; ++i) {
    std::ranges::copy(matrix.begin() + i * original_size, matrix.begin() + (i + 1) * original_size,
                      padded_matrix.begin() + i * new_size);
  }
  return padded_matrix;
}

std::vector<double> nasedkin_e_strassen_algorithm_seq::StrassenSequential::TrimMatrixToOriginalSize(
    const std::vector<double>& matrix, int original_size, int padded_size) {
  std::vector<double> trimmed_matrix(original_size * original_size);
  for (int i = 0; i < original_size; ++i) {
    std::ranges::copy(matrix.begin() + i * padded_size, matrix.begin() + i * padded_size + original_size,
                      trimmed_matrix.begin() + i * original_size);
  }
  return trimmed_matrix;
}

std::vector<double> nasedkin_e_strassen_algorithm_seq::StrassenSequential::StrassenMultiply(
    const std::vector<double>& a, const std::vector<double>& b, int size) {
  if (size <= 32) {
    return StandardMultiply(a, b, size);
  }

  int half_size = size / 2;
  std::vector<double> a11(half_size * half_size);
  std::vector<double> a12(half_size * half_size);
  std::vector<double> a21(half_size * half_size);
  std::vector<double> a22(half_size * half_size);

  std::vector<double> b11(half_size * half_size);
  std::vector<double> b12(half_size * half_size);
  std::vector<double> b21(half_size * half_size);
  std::vector<double> b22(half_size * half_size);

  SplitMatrix(a, a11, 0, 0, size);
  SplitMatrix(a, a12, 0, half_size, size);
  SplitMatrix(a, a21, half_size, 0, size);
  SplitMatrix(a, a22, half_size, half_size, size);

  SplitMatrix(b, b11, 0, 0, size);
  SplitMatrix(b, b12, 0, half_size, size);
  SplitMatrix(b, b21, half_size, 0, size);
  SplitMatrix(b, b22, half_size, half_size, size);

  std::vector<double> p1 =
      StrassenMultiply(AddMatrices(a11, a22, half_size), AddMatrices(b11, b22, half_size), half_size);
  std::vector<double> p2 = StrassenMultiply(AddMatrices(a21, a22, half_size), b11, half_size);
  std::vector<double> p3 = StrassenMultiply(a11, SubtractMatrices(b12, b22, half_size), half_size);
  std::vector<double> p4 = StrassenMultiply(a22, SubtractMatrices(b21, b11, half_size), half_size);
  std::vector<double> p5 = StrassenMultiply(AddMatrices(a11, a12, half_size), b22, half_size);
  std::vector<double> p6 =
      StrassenMultiply(SubtractMatrices(a21, a11, half_size), AddMatrices(b11, b12, half_size), half_size);
  std::vector<double> p7 =
      StrassenMultiply(SubtractMatrices(a12, a22, half_size), AddMatrices(b21, b22, half_size), half_size);

  std::vector<double> c11 = AddMatrices(SubtractMatrices(AddMatrices(p1, p4, half_size), p5, half_size), p7, half_size);
  std::vector<double> c12 = AddMatrices(p3, p5, half_size);
  std::vector<double> c21 = AddMatrices(p2, p4, half_size);
  std::vector<double> c22 = AddMatrices(SubtractMatrices(AddMatrices(p1, p3, half_size), p2, half_size), p6, half_size);

  std::vector<double> result(size * size);
  MergeMatrix(result, c11, 0, 0, size);
  MergeMatrix(result, c12, 0, half_size, size);
  MergeMatrix(result, c21, half_size, 0, size);
  MergeMatrix(result, c22, half_size, half_size, size);

  return result;
}

void nasedkin_e_strassen_algorithm_seq::StrassenSequential::SplitMatrix(const std::vector<double>& parent,
                                                                        std::vector<double>& child, int row_start,
                                                                        int col_start, int parent_size) {
  int child_size = static_cast<int>(std::sqrt(child.size()));
  for (int i = 0; i < child_size; ++i) {
    std::ranges::copy(parent.begin() + (row_start + i) * parent_size + col_start,
                      parent.begin() + (row_start + i) * parent_size + col_start + child_size,
                      child.begin() + i * child_size);
  }
}

void nasedkin_e_strassen_algorithm_seq::StrassenSequential::MergeMatrix(std::vector<double>& parent,
                                                                        const std::vector<double>& child, int row_start,
                                                                        int col_start, int parent_size) {
  int child_size = static_cast<int>(std::sqrt(child.size()));
  for (int i = 0; i < child_size; ++i) {
    std::ranges::copy(child.begin() + i * child_size, child.begin() + (i + 1) * child_size,
                      parent.begin() + (row_start + i) * parent_size + col_start);
  }
}
