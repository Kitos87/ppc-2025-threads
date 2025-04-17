#include "omp/moiseev_a_mult_mat/include/ops_omp.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

bool moiseev_a_mult_mat_omp::MultMatOMP::PreProcessingImpl() {
  unsigned int input_size_a = task_data->inputs_count[0];
  unsigned int input_size_b = task_data->inputs_count[1];

  auto* in_ptr_a = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* in_ptr_b = reinterpret_cast<double*>(task_data->inputs[1]);

  matrix_a_ = std::vector<double>(in_ptr_a, in_ptr_a + input_size_a);
  matrix_b_ = std::vector<double>(in_ptr_b, in_ptr_b + input_size_b);

  unsigned int output_size = task_data->outputs_count[0];
  matrix_c_ = std::vector<double>(output_size, 0.0);

  matrix_size_ = static_cast<int>(std::sqrt(input_size_a));

  block_size_ = static_cast<int>(std::sqrt(matrix_size_));
  if (matrix_size_ % block_size_ != 0) {
    block_size_ = 1;
  }

  num_blocks_ = matrix_size_ / block_size_;

  return true;
}

bool moiseev_a_mult_mat_omp::MultMatOMP::ValidationImpl() {
  return (task_data->inputs_count[0] == task_data->inputs_count[1]) &&
         (task_data->inputs_count[0] == task_data->outputs_count[0]);
}

bool moiseev_a_mult_mat_omp::MultMatOMP::RunImpl() {
#pragma omp parallel for
  for (int i_block = 0; i_block < num_blocks_; ++i_block) {
    for (int j_block = 0; j_block < num_blocks_; ++j_block) {
      for (int s = 0; s < num_blocks_; ++s) {
        int a_block_j = (i_block + s) % num_blocks_;
        int b_block_i = a_block_j;

        int i_start = i_block * block_size_;
        int j_start = j_block * block_size_;
        int a_j_start = a_block_j * block_size_;
        int b_i_start = b_block_i * block_size_;

        for (int i = 0; i < block_size_; ++i) {
          for (int j = 0; j < block_size_; ++j) {
            double sum = 0.0;
            for (int k = 0; k < block_size_; ++k) {
              double a_val = matrix_a_[((i_start + i) * matrix_size_) + (a_j_start + k)];
              double b_val = matrix_b_[((b_i_start + k) * matrix_size_) + (j_start + j)];
              sum += a_val * b_val;
            }
            matrix_c_[((i_start + i) * matrix_size_) + (j_start + j)] += sum;
          }
        }
      }
    }
  }
  return true;
}

bool moiseev_a_mult_mat_omp::MultMatOMP::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  std::ranges::copy(matrix_c_, out_ptr);
  return true;
}
