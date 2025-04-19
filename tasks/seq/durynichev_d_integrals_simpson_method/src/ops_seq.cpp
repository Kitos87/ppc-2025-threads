#include "seq/durynichev_d_integrals_simpson_method/include/ops_seq.hpp"

#include <cmath>
#include <vector>

bool durynichev_d_integrals_simpson_method_seq::SimpsonIntegralSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  boundaries_ = std::vector<double>(in_ptr, in_ptr + input_size);
  n_ = static_cast<int>(boundaries_.back());
  boundaries_.pop_back();
  dim_ = boundaries_.size() / 2;

  result_ = 0.0;
  return true;
}

bool durynichev_d_integrals_simpson_method_seq::SimpsonIntegralSequential::ValidationImpl() {
  return task_data->inputs_count[0] >= 3 && task_data->outputs_count[0] == 1 && (n_ % 2 == 0);
}

bool durynichev_d_integrals_simpson_method_seq::SimpsonIntegralSequential::RunImpl() {
  if (dim_ == 1) {
    result_ = Simpson1D(boundaries_[0], boundaries_[1]);
  } else if (dim_ == 2) {
    result_ = Simpson2D(boundaries_[0], boundaries_[1], boundaries_[2], boundaries_[3]);
  }
  return true;
}

bool durynichev_d_integrals_simpson_method_seq::SimpsonIntegralSequential::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = result_;
  return true;
}

double durynichev_d_integrals_simpson_method_seq::SimpsonIntegralSequential::Func1D(double x) { return x * x; }

double durynichev_d_integrals_simpson_method_seq::SimpsonIntegralSequential::Func2D(double x, double y) {
  return (x * x) + (y * y);
}

double durynichev_d_integrals_simpson_method_seq::SimpsonIntegralSequential::Simpson1D(double a, double b) const {
  double h = (b - a) / n_;
  double sum = Func1D(a) + Func1D(b);

  for (int i = 1; i < n_; i += 2) {
    sum += 4 * Func1D(a + (i * h));
  }
  for (int i = 2; i < n_ - 1; i += 2) {
    sum += 2 * Func1D(a + (i * h));
  }

  return sum * h / 3.0;
}

double durynichev_d_integrals_simpson_method_seq::SimpsonIntegralSequential::Simpson2D(double x0, double x1, double y0,
                                                                                       double y1) const {
  double hx = (x1 - x0) / n_;
  double hy = (y1 - y0) / n_;
  double sum = 0.0;

  for (int i = 0; i <= n_; i++) {
    double x = x0 + (i * hx);
    double coef_x = NAN;

    if (i == 0 || i == n_) {
      coef_x = 1;
    } else if (i % 2 != 0) {
      coef_x = 4;
    } else {
      coef_x = 2;
    }

    for (int j = 0; j <= n_; j++) {
      double y = y0 + (j * hy);
      double coef_y = NAN;

      if (j == 0 || j == n_) {
        coef_y = 1;
      } else if (j % 2 != 0) {
        coef_y = 4;
      } else {
        coef_y = 2;
      }

      sum += coef_x * coef_y * Func2D(x, y);
    }
  }

  return sum * hx * hy / 9.0;
}