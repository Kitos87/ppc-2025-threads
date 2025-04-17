#include "seq/borisov_s_strassen_seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace borisov_s_strassen_seq {

namespace {

std::vector<double> MultiplyNaive(const std::vector<double> &a, const std::vector<double> &b, int n) {
  std::vector<double> c(n * n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int k = 0; k < n; ++k) {
        sum += a[(i * n) + k] * b[(k * n) + j];
      }
      c[(i * n) + j] = sum;
    }
  }
  return c;
}

std::vector<double> AddMatr(const std::vector<double> &a, const std::vector<double> &b, int n) {
  std::vector<double> c(n * n);
  for (int i = 0; i < n * n; ++i) {
    c[i] = a[i] + b[i];
  }
  return c;
}

std::vector<double> SubMatr(const std::vector<double> &a, const std::vector<double> &b, int n) {
  std::vector<double> c(n * n);
  for (int i = 0; i < n * n; ++i) {
    c[i] = a[i] - b[i];
  }
  return c;
}

std::vector<double> SubMatrix(const std::vector<double> &m, int n, int row, int col, int size) {
  std::vector<double> sub(size * size);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      sub[(i * size) + j] = m[((row + i) * n) + (col + j)];
    }
  }
  return sub;
}

void SetSubMatrix(std::vector<double> &m, const std::vector<double> &sub, int n, int row, int col, int size) {
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      m[((row + i) * n) + (col + j)] = sub[(i * size) + j];
    }
  }
}

std::vector<double> StrassenRecursive(const std::vector<double> &a, const std::vector<double> &b, int n) {
  if (n <= 16) {
    return MultiplyNaive(a, b, n);
  }
  int k = n / 2;
  auto a11 = SubMatrix(a, n, 0, 0, k);
  auto a12 = SubMatrix(a, n, 0, k, k);
  auto a21 = SubMatrix(a, n, k, 0, k);
  auto a22 = SubMatrix(a, n, k, k, k);

  auto b11 = SubMatrix(b, n, 0, 0, k);
  auto b12 = SubMatrix(b, n, 0, k, k);
  auto b21 = SubMatrix(b, n, k, 0, k);
  auto b22 = SubMatrix(b, n, k, k, k);

  auto m1 = StrassenRecursive(AddMatr(a11, a22, k), AddMatr(b11, b22, k), k);
  auto m2 = StrassenRecursive(AddMatr(a21, a22, k), b11, k);
  auto m3 = StrassenRecursive(a11, SubMatr(b12, b22, k), k);
  auto m4 = StrassenRecursive(a22, SubMatr(b21, b11, k), k);
  auto m5 = StrassenRecursive(AddMatr(a11, a12, k), b22, k);
  auto m6 = StrassenRecursive(SubMatr(a21, a11, k), AddMatr(b11, b12, k), k);
  auto m7 = StrassenRecursive(SubMatr(a12, a22, k), AddMatr(b21, b22, k), k);

  std::vector<double> c(n * n, 0.0);

  auto c11 = AddMatr(SubMatr(AddMatr(m1, m4, k), m5, k), m7, k);
  auto c12 = AddMatr(m3, m5, k);
  auto c21 = AddMatr(m2, m4, k);
  auto c22 = AddMatr(AddMatr(SubMatr(m1, m2, k), m3, k), m6, k);

  SetSubMatrix(c, c11, n, 0, 0, k);
  SetSubMatrix(c, c12, n, 0, k, k);
  SetSubMatrix(c, c21, n, k, 0, k);
  SetSubMatrix(c, c22, n, k, k, k);

  return c;
}

int NextPowerOfTwo(int n) {
  int r = 1;
  while (r < n) {
    r <<= 1;
  }
  return r;
}

}  // namespace

bool SequentialStrassenSeq::PreProcessingImpl() {
  size_t input_count = task_data->inputs_count[0];
  auto *double_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  input_.assign(double_ptr, double_ptr + input_count);

  size_t output_count = task_data->outputs_count[0];
  output_.resize(output_count, 0.0);

  if (input_.size() < 4) {
    return false;
  }

  rowsA_ = static_cast<int>(input_[0]);
  colsA_ = static_cast<int>(input_[1]);
  rowsB_ = static_cast<int>(input_[2]);
  colsB_ = static_cast<int>(input_[3]);

  return true;
}

bool SequentialStrassenSeq::ValidationImpl() {
  if (colsA_ != rowsB_) {
    return false;
  }

  size_t needed = 4 + (static_cast<size_t>(rowsA_) * colsA_) + (static_cast<size_t>(rowsB_) * colsB_);

  return input_.size() >= needed;
}

bool SequentialStrassenSeq::RunImpl() {
  size_t offset = 4;
  std::vector<double> a(rowsA_ * colsA_);
  for (int i = 0; i < rowsA_ * colsA_; ++i) {
    a[i] = input_[offset + i];
  }
  offset += static_cast<size_t>(rowsA_ * colsA_);

  std::vector<double> b(rowsB_ * colsB_);
  for (int i = 0; i < rowsB_ * colsB_; ++i) {
    b[i] = input_[offset + i];
  }

  int max_dim = std::max({rowsA_, colsA_, colsB_});
  int m = NextPowerOfTwo(max_dim);

  std::vector<double> a_exp(m * m, 0.0);
  std::vector<double> b_exp(m * m, 0.0);

  for (int i = 0; i < rowsA_; ++i) {
    for (int j = 0; j < colsA_; ++j) {
      a_exp[(i * m) + j] = a[(i * colsA_) + j];
    }
  }
  for (int i = 0; i < rowsB_; ++i) {
    for (int j = 0; j < colsB_; ++j) {
      b_exp[(i * m) + j] = b[(i * colsB_) + j];
    }
  }

  auto c_exp = StrassenRecursive(a_exp, b_exp, m);

  std::vector<double> c(rowsA_ * colsB_, 0.0);
  for (int i = 0; i < rowsA_; ++i) {
    for (int j = 0; j < colsB_; ++j) {
      c[(i * colsB_) + j] = c_exp[(i * m) + j];
    }
  }

  output_[0] = static_cast<double>(rowsA_);
  output_[1] = static_cast<double>(colsB_);
  for (int i = 0; i < rowsA_ * colsB_; ++i) {
    output_[2 + i] = c[i];
  }

  return true;
}

bool SequentialStrassenSeq::PostProcessingImpl() {
  auto *out_ptr = reinterpret_cast<double *>(task_data->outputs[0]);
  for (size_t i = 0; i < output_.size(); ++i) {
    out_ptr[i] = output_[i];
  }
  return true;
}

}  // namespace borisov_s_strassen_seq
