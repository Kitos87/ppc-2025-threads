#include "omp/vavilov_v_cannon/include/ops_omp.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

bool vavilov_v_cannon_omp::CannonOMP::PreProcessingImpl() {
  N_ = static_cast<int>(std::sqrt(task_data->inputs_count[0]));
  num_blocks_ = static_cast<int>(task_data->inputs_count[2]);
  block_size_ = N_ / num_blocks_;

  auto* a = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* b = reinterpret_cast<double*>(task_data->inputs[1]);
  A_.assign(a, a + (N_ * N_));
  B_.assign(b, b + (N_ * N_));
  C_.assign(N_ * N_, 0);

  return true;
}

bool vavilov_v_cannon_omp::CannonOMP::ValidationImpl() {
  if (task_data->inputs_count[0] != task_data->inputs_count[1] ||
      task_data->outputs_count[0] != task_data->inputs_count[0]) {
    return false;
  }

  auto n = static_cast<int>(std::sqrt(task_data->inputs_count[0]));
  auto num_blocks = static_cast<int>(task_data->inputs_count[2]);
  return n % num_blocks == 0;
}

void vavilov_v_cannon_omp::CannonOMP::InitialShift() {
  std::vector<double> a_tmp = A_;
  std::vector<double> b_tmp = B_;

#pragma omp parallel for
  for (int bi = 0; bi < num_blocks_; ++bi) {
    for (int bj = 0; bj < num_blocks_; ++bj) {
      int src_row = (bi + bj) % num_blocks_;
      int src_col = (bj + bi) % num_blocks_;
      for (int i = 0; i < block_size_; ++i) {
        for (int j = 0; j < block_size_; ++j) {
          B_[(((bi * block_size_) + i) * N_) + ((bj * block_size_) + j)] =
              b_tmp[(((src_row * block_size_) + i) * N_) + ((bj * block_size_) + j)];
          A_[(((bi * block_size_) + i) * N_) + ((bj * block_size_) + j)] =
              a_tmp[(((bi * block_size_) + i) * N_) + ((src_col * block_size_) + j)];
        }
      }
    }
  }
}
void vavilov_v_cannon_omp::CannonOMP::BlockMultiply() {
#pragma omp parallel for
  for (int bi = 0; bi < num_blocks_; ++bi) {
    for (int bj = 0; bj < num_blocks_; ++bj) {
      for (int i = 0; i < block_size_; ++i) {
        for (int j = 0; j < block_size_; ++j) {
          double temp = 0.0;
          for (int k = 0; k < block_size_; ++k) {
            int row = (bi * block_size_) + i;
            int col = (bj * block_size_) + j;
            int k_idx = (bj * block_size_) + k;
            int k_row = (bi * block_size_) + k;
            temp += A_[(row * N_) + k_idx] * B_[(k_row * N_) + col];
          }
          C_[(((bi * block_size_) + i) * N_) + ((bj * block_size_) + j)] += temp;
        }
      }
    }
  }
}

void vavilov_v_cannon_omp::CannonOMP::ShiftBlocks() {
  std::vector<double> a_tmp = A_;
  std::vector<double> b_tmp = B_;

#pragma omp parallel for
  for (int bi = 0; bi < num_blocks_; ++bi) {
    for (int bj = 0; bj < num_blocks_; ++bj) {
      int src_row = (bi + 1) % num_blocks_;
      int src_col = (bj + 1) % num_blocks_;
      for (int i = 0; i < block_size_; ++i) {
        for (int j = 0; j < block_size_; ++j) {
          B_[(((bi * block_size_) + i) * N_) + ((bj * block_size_) + j)] =
              b_tmp[(((src_row * block_size_) + i) * N_) + ((bj * block_size_) + j)];
          A_[(((bi * block_size_) + i) * N_) + ((bj * block_size_) + j)] =
              a_tmp[(((bi * block_size_) + i) * N_) + ((src_col * block_size_) + j)];
        }
      }
    }
  }
}

bool vavilov_v_cannon_omp::CannonOMP::RunImpl() {
  InitialShift();
  for (int iter = 0; iter < num_blocks_; ++iter) {
    BlockMultiply();
    ShiftBlocks();
  }
  return true;
}

bool vavilov_v_cannon_omp::CannonOMP::PostProcessingImpl() {
  std::ranges::copy(C_, reinterpret_cast<double*>(task_data->outputs[0]));
  return true;
}
