#include "seq/lysov_i_matrix_multiplication_Fox_algorithm/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

void lysov_i_matrix_multiplication_fox_algorithm_seq::ProcessBlock(const std::vector<double> &a,
                                                                   const std::vector<double> &b, std::vector<double> &c,
                                                                   std::size_t i, std::size_t j,
                                                                   std::size_t a_block_row, std::size_t block_size,
                                                                   std::size_t n) {
  std::size_t block_h = std::min(block_size, n - (i * block_size));
  std::size_t block_w = std::min(block_size, n - (j * block_size));
  for (std::size_t ii = 0; ii < block_h; ++ii) {
    for (std::size_t jj = 0; jj < block_w; ++jj) {
      double sum = 0.0;
      for (std::size_t kk = 0; kk < std::min(block_size, n - (a_block_row * block_size)); ++kk) {
        std::size_t row_a = (i * block_size) + ii;
        std::size_t col_a = (a_block_row * block_size) + kk;
        std::size_t row_b = (a_block_row * block_size) + kk;
        std::size_t col_b = (j * block_size) + jj;
        if (row_a < n && col_a < n && row_b < n && col_b < n) {
          sum += a[(row_a * n) + col_a] * b[(row_b * n) + col_b];
        }
      }
      std::size_t row_c = (i * block_size) + ii;
      std::size_t col_c = (j * block_size) + jj;
      if (row_c < n && col_c < n) {
        c[(row_c * n) + col_c] += sum;
      }
    }
  }
}

bool lysov_i_matrix_multiplication_fox_algorithm_seq::TestTaskSequential::PreProcessingImpl() {
  n_ = reinterpret_cast<std::size_t *>(task_data->inputs[0])[0];
  block_size_ = reinterpret_cast<std::size_t *>(task_data->inputs[3])[0];
  a_.resize(n_ * n_);
  b_.resize(n_ * n_);
  c_.resize(n_ * n_, 0.0);
  std::copy(reinterpret_cast<double *>(task_data->inputs[1]),
            reinterpret_cast<double *>(task_data->inputs[1]) + (n_ * n_), a_.begin());
  std::copy(reinterpret_cast<double *>(task_data->inputs[2]),
            reinterpret_cast<double *>(task_data->inputs[2]) + (n_ * n_), b_.begin());
  return true;
}

bool lysov_i_matrix_multiplication_fox_algorithm_seq::TestTaskSequential::ValidationImpl() {
  n_ = reinterpret_cast<std::size_t *>(task_data->inputs[0])[0];
  block_size_ = reinterpret_cast<std::size_t *>(task_data->inputs[3])[0];
  if (task_data->inputs_count.size() != 3 || task_data->outputs_count.size() != 1) {
    return false;
  }
  if (task_data->inputs_count[1] != n_ * n_ || task_data->inputs_count[0] != n_ * n_) {
    return false;
  }
  return task_data->outputs_count[0] == n_ * n_ && block_size_ > 0;
}

bool lysov_i_matrix_multiplication_fox_algorithm_seq::TestTaskSequential::RunImpl() {
  std::size_t num_blocks = (n_ + block_size_ - 1) / block_size_;
  for (std::size_t step = 0; step < num_blocks; ++step) {
    for (std::size_t i = 0; i < num_blocks; ++i) {
      std::size_t a_block_row = (i + step) % num_blocks;
      for (std::size_t j = 0; j < num_blocks; ++j) {
        ProcessBlock(a_, b_, c_, i, j, a_block_row, block_size_, n_);
      }
    }
  }
  return true;
}

bool lysov_i_matrix_multiplication_fox_algorithm_seq::TestTaskSequential::PostProcessingImpl() {
  std::ranges::copy(c_, reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}
