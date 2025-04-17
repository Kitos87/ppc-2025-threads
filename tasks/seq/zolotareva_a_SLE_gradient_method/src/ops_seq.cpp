#include "seq/zolotareva_a_SLE_gradient_method/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

bool zolotareva_a_sle_gradient_method_seq::TestTaskSequential::PreProcessingImpl() {
  n_ = static_cast<int>(task_data->inputs_count[1]);
  a_.resize(n_ * n_);
  b_.resize(n_);
  x_.resize(n_, 0.0);
  const auto* input_matrix = reinterpret_cast<const double*>(task_data->inputs[0]);
  const auto* input_vector = reinterpret_cast<const double*>(task_data->inputs[1]);

  for (int i = 0; i < n_; ++i) {
    b_[i] = input_vector[i];
    for (int j = 0; j < n_; ++j) {
      a_[(i * n_) + j] = input_matrix[(i * n_) + j];
    }
  }

  return true;
}

bool zolotareva_a_sle_gradient_method_seq::TestTaskSequential::ValidationImpl() {
  if (static_cast<int>(task_data->inputs_count[0]) < 0 || static_cast<int>(task_data->inputs_count[1]) < 0 ||
      static_cast<int>(task_data->outputs_count[0]) < 0) {
    return false;
  }
  if (task_data->inputs_count.size() < 2 || task_data->inputs.size() < 2 || task_data->outputs.empty()) {
    return false;
  }

  if (static_cast<int>(task_data->inputs_count[0]) !=
      (static_cast<int>(task_data->inputs_count[1]) * static_cast<int>(task_data->inputs_count[1]))) {
    return false;
  }
  if (task_data->outputs_count[0] != task_data->inputs_count[1]) {
    return false;
  }

  // проверка симметрии и положительной определённости
  const auto* a = reinterpret_cast<const double*>(task_data->inputs[0]);

  return IsPositiveAndSimm(a, static_cast<int>(task_data->inputs_count[1]));
}

bool zolotareva_a_sle_gradient_method_seq::TestTaskSequential::RunImpl() {
  ConjugateGradient(a_, b_, x_, n_);
  return true;
}

bool zolotareva_a_sle_gradient_method_seq::TestTaskSequential::PostProcessingImpl() {
  auto* output_raw = reinterpret_cast<double*>(task_data->outputs[0]);
  std::ranges::copy(x_.begin(), x_.end(), output_raw);
  return true;
}

void zolotareva_a_sle_gradient_method_seq::TestTaskSequential::ConjugateGradient(const std::vector<double>& a,
                                                                                 const std::vector<double>& b,
                                                                                 std::vector<double>& x, int n) {
  double initial_res_norm = 0.0;
  DotProduct(initial_res_norm, b, b, n);
  initial_res_norm = std::sqrt(initial_res_norm);
  double threshold = initial_res_norm == 0.0 ? 1e-12 : (1e-12 * initial_res_norm);

  std::vector<double> r = b;  // начальный вектор невязки r = b - a*x0, x0 = 0
  std::vector<double> p = r;  // начальное направление поиска p = r
  double rs_old = 0;
  DotProduct(rs_old, r, r, n);

  for (int s = 0; s <= n; ++s) {
    std::vector<double> ap(n, 0.0);
    MatrixVectorMult(a, p, ap, n);
    double p_ap = 0.0;
    DotProduct(p_ap, p, ap, n);
    if (p_ap == 0.0) {
      break;
    }

    double alpha = rs_old / p_ap;

    for (int i = 0; i < n; ++i) {
      x[i] += alpha * p[i];
      r[i] -= alpha * ap[i];
    }

    double rs_new = 0.0;
    DotProduct(rs_new, r, r, n);
    if (rs_new < threshold) {  // Проверка на сходимость
      break;
    }
    double beta = rs_new / rs_old;
    for (int i = 0; i < n; ++i) {
      p[i] = r[i] + beta * p[i];
    }

    rs_old = rs_new;
  }
}

void zolotareva_a_sle_gradient_method_seq::TestTaskSequential::DotProduct(double& sum, const std::vector<double>& vec1,
                                                                          const std::vector<double>& vec2, int n) {
  for (int i = 0; i < n; ++i) {
    sum += vec1[i] * vec2[i];
  }
}

void zolotareva_a_sle_gradient_method_seq::TestTaskSequential::MatrixVectorMult(const std::vector<double>& matrix,
                                                                                const std::vector<double>& vector,
                                                                                std::vector<double>& result, int n) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      result[i] += matrix[(i * n) + j] * vector[j];
    }
  }
}

bool zolotareva_a_sle_gradient_method_seq::TestTaskSequential::IsPositiveAndSimm(const double* a, int n) {
  std::vector<double> m(n * n);
  // копируем и проверяем симметричность
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double val = a[(i * n) + j];
      m[(i * n) + j] = val;
      if (j > i) {
        if (val != a[(j * n) + i]) {
          return false;
        }
      }
    }
  }
  // проверяем позитивную определенность
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      double sum = m[(i * n) + j];
      for (int k = 0; k < j; k++) {
        sum -= m[(i * n) + k] * m[(j * n) + k];
      }

      if (i == j) {
        if (sum <= 1e-15) {
          return false;
        }
        m[(i * n) + j] = std::sqrt(sum);
      } else {
        m[(i * n) + j] = sum / m[(j * n) + j];
      }
    }
  }
  return true;
}
