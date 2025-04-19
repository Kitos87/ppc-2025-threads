#include "seq/solovev_a_ccs_mmult_sparse/include/ccs_mmult_sparse.hpp"

#include <complex>
#include <vector>

bool solovev_a_matrix::SeqMatMultCcs::PreProcessingImpl() {
  M1_ = reinterpret_cast<MatrixInCcsSparse*>(task_data->inputs[0]);
  M2_ = reinterpret_cast<MatrixInCcsSparse*>(task_data->inputs[1]);
  M3_ = reinterpret_cast<MatrixInCcsSparse*>(task_data->outputs[0]);
  return true;
}

bool solovev_a_matrix::SeqMatMultCcs::ValidationImpl() {
  int m1_c_n = reinterpret_cast<MatrixInCcsSparse*>(task_data->inputs[0])->c_n;
  int m2_r_n = reinterpret_cast<MatrixInCcsSparse*>(task_data->inputs[1])->r_n;
  return (m1_c_n == m2_r_n);
}

bool solovev_a_matrix::SeqMatMultCcs::RunImpl() {
  M3_->r_n = M1_->r_n;
  M3_->c_n = M2_->c_n;
  M3_->col_p.resize(M3_->c_n + 1);
  M3_->col_p[0] = 0;
  std::vector<int> available_el(M3_->r_n);
  for (int m2_c = 0; m2_c < M3_->c_n; ++m2_c) {
    for (int m3_r = 0; m3_r < M3_->r_n; ++m3_r) {
      available_el[m3_r] = 0;
    }
    for (int m2_i = M2_->col_p[m2_c]; m2_i < M2_->col_p[m2_c + 1]; ++m2_i) {
      int m2_r = M2_->row[m2_i];
      for (int m1_i = M1_->col_p[m2_r]; m1_i < M1_->col_p[m2_r + 1]; ++m1_i) {
        available_el[M1_->row[m1_i]] = 1;
      }
    }
    int n_z_c_cnt = 0;
    for (int m3_r = 0; m3_r < M3_->r_n; ++m3_r) {
      n_z_c_cnt += available_el[m3_r];
    }
    M3_->col_p[m2_c + 1] = n_z_c_cnt + M3_->col_p[m2_c];
  }

  int n_z_full = M3_->col_p[M3_->c_n];
  M3_->n_z = n_z_full;
  M3_->row.resize(n_z_full);
  M3_->val.resize(n_z_full);

  std::complex<double> nought = {0.0, 0.0};
  std::complex<double> m2_val = {0.0, 0.0};
  std::vector<std::complex<double>> cask(M3_->r_n);
  for (int m2_c = 0; m2_c < M3_->c_n; ++m2_c) {
    for (int m3_r = 0; m3_r < M3_->r_n; ++m3_r) {
      cask[m3_r] = nought;
      available_el[m3_r] = 0;
    }

    for (int m2_i = M2_->col_p[m2_c]; m2_i < M2_->col_p[m2_c + 1]; ++m2_i) {
      int m2_r = M2_->row[m2_i];
      m2_val = M2_->val[m2_i];
      for (int m1_i = M1_->col_p[m2_r]; m1_i < M1_->col_p[m2_r + 1]; ++m1_i) {
        int m1_row = M1_->row[m1_i];
        cask[m1_row] += M1_->val[m1_i] * m2_val;
        available_el[m1_row] = 1;
      }
    }

    int c_pos = M3_->col_p[m2_c];
    for (int m3_r = 0; m3_r < M3_->r_n; ++m3_r) {
      if (available_el[m3_r] != 0) {
        M3_->row[c_pos] = m3_r;
        M3_->val[c_pos++] = cask[m3_r];
      }
    }
  }

  return true;
}

bool solovev_a_matrix::SeqMatMultCcs::PostProcessingImpl() { return true; }
