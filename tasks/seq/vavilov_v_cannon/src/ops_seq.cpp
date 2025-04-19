#include "seq/vavilov_v_cannon/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

bool vavilov_v_cannon_seq::CannonSequential::PreProcessingImpl() {
  N_ = static_cast<unsigned int>(std::sqrt(task_data->inputs_count[0]));
  num_blocks_ = static_cast<unsigned int>(task_data->inputs_count[2]);
  block_size_ = N_ / num_blocks_;

  auto* a = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* b = reinterpret_cast<double*>(task_data->inputs[1]);
  A_.assign(a, a + (N_ * N_));
  B_.assign(b, b + (N_ * N_));
  C_.assign(N_ * N_, 0);

  return true;
}

bool vavilov_v_cannon_seq::CannonSequential::ValidationImpl() {
  if (task_data->inputs_count[0] != task_data->inputs_count[1] ||
      task_data->outputs_count[0] != task_data->inputs_count[0]) {
    return false;
  }

  auto n = static_cast<unsigned int>(std::sqrt(task_data->inputs_count[0]));
  auto num_blocks = static_cast<unsigned int>(task_data->inputs_count[2]);
  return n % num_blocks == 0;
}

void vavilov_v_cannon_seq::CannonSequential::InitialShift() {
  std::vector<double> a_tmp = A_;
  std::vector<double> b_tmp = B_;

  for (unsigned int bi = 0; bi < num_blocks_; ++bi) {
    for (unsigned int bj = 0; bj < num_blocks_; ++bj) {
      unsigned int src_row = (bi + bj) % num_blocks_;
      unsigned int src_col = (bj + bi) % num_blocks_;
      for (unsigned int i = 0; i < block_size_; ++i) {
        for (unsigned int j = 0; j < block_size_; ++j) {
          B_[(((bi * block_size_) + i) * N_) + ((bj * block_size_) + j)] =
              b_tmp[(((src_row * block_size_) + i) * N_) + ((bj * block_size_) + j)];
          A_[(((bi * block_size_) + i) * N_) + ((bj * block_size_) + j)] =
              a_tmp[(((bi * block_size_) + i) * N_) + ((src_col * block_size_) + j)];
        }
      }
    }
  }
}

void vavilov_v_cannon_seq::CannonSequential::BlockMultiply() {
  for (unsigned int bi = 0; bi < N_; bi += block_size_) {
    for (unsigned int bj = 0; bj < N_; bj += block_size_) {
      for (unsigned int i = bi; i < bi + block_size_; i++) {
        for (unsigned int j = bj; j < bj + block_size_; j++) {
          double temp = 0.0;
          for (unsigned int k = 0; k < block_size_; k++) {
            unsigned int row_a = bi + (i - bi);
            unsigned int col_a = bj + k;
            unsigned int row_b = bi + k;
            unsigned int col_b = bj + (j - bj);

            temp += A_[(row_a * N_) + col_a] * B_[(row_b * N_) + col_b];
          }

          C_[(i * N_) + j] += temp;
        }
      }
    }
  }
}

void vavilov_v_cannon_seq::CannonSequential::ShiftBlocks() {
  std::vector<double> a_tmp = A_;
  std::vector<double> b_tmp = B_;

  for (unsigned int bi = 0; bi < num_blocks_; ++bi) {
    for (unsigned int bj = 0; bj < num_blocks_; ++bj) {
      unsigned int src_row = (bi + 1) % num_blocks_;
      unsigned int src_col = (bj + 1) % num_blocks_;
      for (unsigned int i = 0; i < block_size_; ++i) {
        for (unsigned int j = 0; j < block_size_; ++j) {
          B_[(((bi * block_size_) + i) * N_) + ((bj * block_size_) + j)] =
              b_tmp[(((src_row * block_size_) + i) * N_) + ((bj * block_size_) + j)];
          A_[(((bi * block_size_) + i) * N_) + ((bj * block_size_) + j)] =
              a_tmp[(((bi * block_size_) + i) * N_) + ((src_col * block_size_) + j)];
        }
      }
    }
  }
}

bool vavilov_v_cannon_seq::CannonSequential::RunImpl() {
  InitialShift();
  for (unsigned int iter = 0; iter < num_blocks_; ++iter) {
    BlockMultiply();
    ShiftBlocks();
  }
  return true;
}

bool vavilov_v_cannon_seq::CannonSequential::PostProcessingImpl() {
  std::ranges::copy(C_, reinterpret_cast<double*>(task_data->outputs[0]));
  return true;
}
