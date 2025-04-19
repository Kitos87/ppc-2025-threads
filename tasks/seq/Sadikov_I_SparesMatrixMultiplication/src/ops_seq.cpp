#include "seq/Sadikov_I_SparesMatrixMultiplication/include/ops_seq.hpp"

#include <algorithm>
#include <random>
#include <vector>

#include "seq/Sadikov_I_SparesMatrixMultiplication/include/SparesMatrix.hpp"

bool sadikov_i_sparse_matrix_multiplication_task_seq::CCSMatrixSequential::PreProcessingImpl() {
  auto fmatrix_rows_count = static_cast<int>(task_data->inputs_count[0]);
  auto fmatrxix_columns_count = static_cast<int>(task_data->inputs_count[1]);
  auto smatrix_rows_count = static_cast<int>(task_data->inputs_count[2]);
  auto smatrix_columns_count = static_cast<int>(task_data->inputs_count[3]);
  if (fmatrix_rows_count == 0 || fmatrxix_columns_count == 0 || smatrix_rows_count == 0 || smatrix_columns_count == 0) {
    return true;
  }
  std::vector<double> fmatrix;
  fmatrix.reserve(fmatrix_rows_count * fmatrxix_columns_count);
  std::vector<double> smatrix;
  smatrix.reserve(smatrix_rows_count * smatrix_columns_count);
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  for (auto i = 0; i < fmatrix_rows_count * fmatrxix_columns_count; ++i) {
    fmatrix.emplace_back(in_ptr[i]);
  }
  m_fMatrix_ = MatrixToSpares(fmatrix_rows_count, fmatrxix_columns_count, fmatrix);
  auto *in_ptr2 = reinterpret_cast<double *>(task_data->inputs[1]);
  for (auto i = 0; i < smatrix_rows_count * smatrix_columns_count; ++i) {
    smatrix.emplace_back(in_ptr2[i]);
  }
  m_sMatrix_ = MatrixToSpares(smatrix_rows_count, smatrix_columns_count, smatrix);
  return true;
}

bool sadikov_i_sparse_matrix_multiplication_task_seq::CCSMatrixSequential::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->inputs_count[3] &&
         task_data->inputs_count[1] == task_data->inputs_count[2];
}

bool sadikov_i_sparse_matrix_multiplication_task_seq::CCSMatrixSequential::RunImpl() {
  m_answerMatrix_ = m_fMatrix_ * m_sMatrix_;
  return true;
}

bool sadikov_i_sparse_matrix_multiplication_task_seq::CCSMatrixSequential::PostProcessingImpl() {
  auto answer = FromSparesMatrix(m_answerMatrix_);
  for (auto i = 0; i < static_cast<int>(answer.size()); ++i) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = answer[i];
  }
  return true;
}

std::vector<double> sadikov_i_sparse_matrix_multiplication_task_seq::GetRandomMatrix(int size) {
  std::vector<double> data(size);
  std::random_device dev;
  std::mt19937 gen(dev());
  int high = 500;
  int low = 0;
  std::uniform_int_distribution<> number(low, high);
  for (int i = 0; i < size / 5; ++i) {
    data[i] = static_cast<double>(number(gen));
  }
  std::ranges::shuffle(data, gen);
  return data;
}