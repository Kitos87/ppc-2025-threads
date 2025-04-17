#include "seq/odintsov_m_multmatrix_cannon/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>
using namespace std;
void odintsov_m_mulmatrix_cannon_seq::MulMatrixCannonSequential::ShiftRow(std::vector<double>& matrix, int root,
                                                                          int row, int shift) {
  shift = shift % root;
  std::vector<double> tmp(root);
  for (int j = 0; j < root; j++) {
    tmp[j] = matrix[(row * root) + ((j + shift) % root)];
  }
  for (int j = 0; j < root; j++) {
    matrix[(row * root) + j] = tmp[j];
  }
}

void odintsov_m_mulmatrix_cannon_seq::MulMatrixCannonSequential::ShiftColumn(std::vector<double>& matrix, int root,
                                                                             int col, int shift) {
  shift = shift % root;
  std::vector<double> tmp(root);

  for (int i = 0; i < root; i++) {
    tmp[i] = matrix[(((i + shift) % root) * root) + col];
  }
  for (int i = 0; i < root; i++) {
    matrix[(i * root) + col] = tmp[i];
  }
}
void odintsov_m_mulmatrix_cannon_seq::MulMatrixCannonSequential::ShiftBlocksUp(std::vector<double>& matrix, int root,
                                                                               int sz) const {
  int p = root / block_sz_;
  for (int bj = 0; bj < p; bj++) {
    std::vector<double> first_block(block_sz_ * block_sz_);

    for (int i = 0; i < block_sz_; i++) {
      for (int j = 0; j < block_sz_; j++) {
        first_block[(i * block_sz_) + j] = matrix[(i * root) + ((bj * block_sz_) + j)];
      }
    }

    for (int bi = 0; bi < (p - 1); bi++) {
      for (int i = 0; i < block_sz_; i++) {
        for (int j = 0; j < block_sz_; j++) {
          matrix[((bi * block_sz_ + i) * root) + (bj * block_sz_) + j] =
              matrix[(((bi + 1) * block_sz_) * root) + (i * root) + ((bj * block_sz_) + j)];
        }
      }
    }

    for (int i = 0; i < block_sz_; i++) {
      for (int j = 0; j < block_sz_; j++) {
        matrix[((((p - 1) * block_sz_) * root) + (i * root) + ((bj * block_sz_) + j))] =
            first_block[(i * block_sz_) + j];
      }
    }
  }
}

void odintsov_m_mulmatrix_cannon_seq::MulMatrixCannonSequential::ShiftBlocksLeft(std::vector<double>& matrix, int root,
                                                                                 int sz) const {
  int p = root / block_sz_;
  for (int bi = 0; bi < p; bi++) {
    std::vector<double> first_block(block_sz_ * block_sz_);

    for (int i = 0; i < block_sz_; i++) {
      for (int j = 0; j < block_sz_; j++) {
        first_block[(i * block_sz_) + j] = matrix[((bi * block_sz_ + i) * root) + j];
      }
    }

    for (int bj = 0; bj < (p - 1); bj++) {
      for (int i = 0; i < block_sz_; i++) {
        for (int j = 0; j < block_sz_; j++) {
          matrix[((bi * block_sz_ + i) * root) + (bj * block_sz_) + j] =
              matrix[((bi * block_sz_ + i) * root) + (((bj + 1) * block_sz_) + j)];
        }
      }
    }

    for (int i = 0; i < block_sz_; i++) {
      for (int j = 0; j < block_sz_; j++) {
        matrix[((bi * block_sz_ + i) * root) + (((p - 1) * block_sz_) + j)] = first_block[(i * block_sz_) + j];
      }
    }
  }
}

bool odintsov_m_mulmatrix_cannon_seq::MulMatrixCannonSequential::IsSquere(unsigned int num) {
  auto root = static_cast<unsigned int>(std::sqrt(num));
  return (root * root) == num;
}

int odintsov_m_mulmatrix_cannon_seq::MulMatrixCannonSequential::GetBlockSize(int n) {
  for (int k = (n / 2); k >= 2; k--) {
    if ((n % k) == 0) {
      return k;
    }
  }
  return 1;
}
void odintsov_m_mulmatrix_cannon_seq::MulMatrixCannonSequential::CopyBlock(const std::vector<double>& matrix,
                                                                           std::vector<double>& block, int start,
                                                                           int root, int block_sz) {
  for (int i = 0; i < block_sz; i++) {
    for (int j = 0; j < block_sz; j++) {
      int index = start + (i * root) + j;
      block[(i * block_sz) + j] = matrix[index];
    }
  }
}
void odintsov_m_mulmatrix_cannon_seq::MulMatrixCannonSequential::InitializeShift(std::vector<double>& matrix, int root,
                                                                                 int grid_size, int block_sz,
                                                                                 bool is_row_shift) {
  for (int b = 0; b < grid_size; ++b) {
    for (int index = b * block_sz; index < (b + 1) * block_sz; ++index) {
      for (int shift = 0; shift < b; ++shift) {
        if (is_row_shift) {
          ShiftRow(matrix, root, index, block_sz);
        } else {
          ShiftColumn(matrix, root, index, block_sz);
        }
      }
    }
  }
}
bool odintsov_m_mulmatrix_cannon_seq::MulMatrixCannonSequential::PreProcessingImpl() {
  szA_ = task_data->inputs_count[0];
  szB_ = task_data->inputs_count[1];
  matrixA_.assign(reinterpret_cast<double*>(task_data->inputs[0]),
                  reinterpret_cast<double*>(task_data->inputs[0]) + szA_);
  matrixB_.assign(reinterpret_cast<double*>(task_data->inputs[1]),
                  reinterpret_cast<double*>(task_data->inputs[1]) + szB_);
  matrixC_.assign(szA_, 0);

  block_sz_ = GetBlockSize(static_cast<int>(sqrt(szA_)));
  return true;
}

bool odintsov_m_mulmatrix_cannon_seq::MulMatrixCannonSequential::ValidationImpl() {
  if (task_data->inputs_count[0] != task_data->inputs_count[1]) {
    return false;
  }

  if ((!(IsSquere(task_data->inputs_count[0]))) || (!(IsSquere(task_data->inputs_count[1])))) {
    return false;
  }
  return true;
}

bool odintsov_m_mulmatrix_cannon_seq::MulMatrixCannonSequential::RunImpl() {
  int root = static_cast<int>(sqrt(szA_));

  std::vector<double> block_a(block_sz_ * block_sz_, 0);
  std::vector<double> block_b(block_sz_ * block_sz_, 0);

  int grid_size = root / block_sz_;

  InitializeShift(matrixA_, root, grid_size, block_sz_, true);
  InitializeShift(matrixB_, root, grid_size, block_sz_, false);

  for (int step = 0; step < grid_size; step++) {
    for (int bi = 0; bi < root / block_sz_; bi++) {
      for (int bj = 0; bj < root / block_sz_; bj++) {
        int start = ((bi * block_sz_) * root) + (bj * block_sz_);

        CopyBlock(matrixA_, block_a, start, root, block_sz_);
        CopyBlock(matrixB_, block_b, start, root, block_sz_);

        for (int i = 0; i < block_sz_; i++) {
          for (int k = 0; k < block_sz_; k++) {
            double a_ik = block_a[(i * block_sz_) + k];
            for (int j = 0; j < block_sz_; j++) {
              int index = (((bi * block_sz_) + i) * root) + ((bj * block_sz_) + j);
              matrixC_[index] += a_ik * block_b[(k * block_sz_) + j];
            }
          }
        }
      }
    }

    ShiftBlocksLeft(matrixA_, root, block_sz_);
    ShiftBlocksUp(matrixB_, root, block_sz_);
  }

  return true;
}

bool odintsov_m_mulmatrix_cannon_seq::MulMatrixCannonSequential::PostProcessingImpl() {
  std::size_t sz_c = matrixC_.size();
  for (std::size_t i = 0; i < sz_c; i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = matrixC_[i];
  }
  return true;
}
