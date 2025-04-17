#include "seq/moiseev_a_mult_mat/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

bool moiseev_a_mult_mat_seq::MultMatSequential::PreProcessingImpl() {
  unsigned int input_size_a = task_data->inputs_count[0];
  unsigned int input_size_b = task_data->inputs_count[1];

  auto *in_ptr_a = reinterpret_cast<double *>(task_data->inputs[0]);
  auto *in_ptr_b = reinterpret_cast<double *>(task_data->inputs[1]);

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

bool moiseev_a_mult_mat_seq::MultMatSequential::ValidationImpl() {
  return (task_data->inputs_count[0] == task_data->inputs_count[1]) &&
         (task_data->inputs_count[0] == task_data->outputs_count[0]);
}

bool moiseev_a_mult_mat_seq::MultMatSequential::RunImpl() {
  for (int block_row = 0; block_row < num_blocks_; ++block_row) {
    for (int block_col = 0; block_col < num_blocks_; ++block_col) {
      for (int block_step = 0; block_step < num_blocks_; ++block_step) {
        int a_block_col = (block_row + block_step) % num_blocks_;
        int b_block_row = a_block_col;

        int block_row_start = block_row * block_size_;
        int block_col_start = block_col * block_size_;
        int a_col_start = a_block_col * block_size_;
        int b_row_start = b_block_row * block_size_;

        for (int row_offset = 0; row_offset < block_size_; ++row_offset) {
          for (int col_offset = 0; col_offset < block_size_; ++col_offset) {
            double block_result = 0.0;
            for (int inner_dim_offset = 0; inner_dim_offset < block_size_; ++inner_dim_offset) {
              double a_element =
                  matrix_a_[((block_row_start + row_offset) * matrix_size_) + (a_col_start + inner_dim_offset)];
              double b_element =
                  matrix_b_[((b_row_start + inner_dim_offset) * matrix_size_) + (block_col_start + col_offset)];
              block_result += a_element * b_element;
            }
            matrix_c_[((block_row_start + row_offset) * matrix_size_) + (block_col_start + col_offset)] += block_result;
          }
        }
      }
    }
  }
  return true;
}

bool moiseev_a_mult_mat_seq::MultMatSequential::PostProcessingImpl() {
  auto *out_ptr = reinterpret_cast<double *>(task_data->outputs[0]);
  std::ranges::copy(matrix_c_, out_ptr);
  return true;
}
