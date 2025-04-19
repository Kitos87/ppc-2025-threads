#include "seq/anufriev_d_integrals_simpson/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace {

int SimpsonCoeff(int i, int n) {
  if (i == 0 || i == n) {
    return 1;
  }
  if (i % 2 != 0) {
    return 4;
  }
  return 2;
}
}  // namespace

namespace anufriev_d_integrals_simpson_seq {

double IntegralsSimpsonSequential::FunctionN(const std::vector<double>& coords) const {
  switch (func_code_) {
    case 0: {
      double s = 0.0;
      for (double c : coords) {
        s += c * c;
      }
      return s;
    }
    case 1: {
      double val = 1.0;
      for (size_t i = 0; i < coords.size(); i++) {
        if (i % 2 == 0) {
          val *= std::sin(coords[i]);
        } else {
          val *= std::cos(coords[i]);
        }
      }
      return val;
    }
    default:
      return 0.0;
  }
}

double IntegralsSimpsonSequential::RecursiveSimpsonSum(int dim_index, std::vector<int>& idx,
                                                       const std::vector<double>& steps) const {
  if (dim_index == dimension_) {
    double coeff = 1.0;
    std::vector<double> coords(dimension_);
    for (int d = 0; d < dimension_; ++d) {
      coords[d] = a_[d] + idx[d] * steps[d];
      coeff *= SimpsonCoeff(idx[d], n_[d]);
    }
    return coeff * FunctionN(coords);
  }
  double sum = 0.0;
  for (int i = 0; i <= n_[dim_index]; ++i) {
    idx[dim_index] = i;
    sum += RecursiveSimpsonSum(dim_index + 1, idx, steps);
  }
  return sum;
}

bool IntegralsSimpsonSequential::PreProcessingImpl() {
  if (task_data->inputs.empty()) {
    return false;
  }

  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  size_t in_size_bytes = task_data->inputs_count[0];
  size_t num_doubles = in_size_bytes / sizeof(double);

  if (num_doubles < 1) {
    return false;
  }

  int d = static_cast<int>(in_ptr[0]);
  if (d < 1) {
    return false;
  }

  size_t needed_count = static_cast<size_t>(3 * d) + 2;
  if (num_doubles < needed_count) {
    return false;
  }

  dimension_ = d;
  a_.resize(dimension_);
  b_.resize(dimension_);
  n_.resize(dimension_);

  int idx_ptr = 1;
  for (int i = 0; i < dimension_; i++) {
    a_[i] = in_ptr[idx_ptr++];
    b_[i] = in_ptr[idx_ptr++];
    n_[i] = static_cast<int>(in_ptr[idx_ptr++]);
    if (n_[i] <= 0 || (n_[i] % 2) != 0) {
      return false;
    }
  }

  func_code_ = static_cast<int>(in_ptr[idx_ptr]);

  result_ = 0.0;

  return true;
}

bool IntegralsSimpsonSequential::ValidationImpl() {
  if (task_data->outputs.empty()) {
    return false;
  }
  if (task_data->outputs_count.empty() || task_data->outputs_count[0] < 1) {
    return false;
  }
  return true;
}

bool IntegralsSimpsonSequential::RunImpl() {
  std::vector<double> steps(dimension_);
  for (int i = 0; i < dimension_; i++) {
    steps[i] = (b_[i] - a_[i]) / n_[i];
  }

  std::vector<int> idx(dimension_, 0);
  double sum = RecursiveSimpsonSum(0, idx, steps);

  double coeff = 1.0;
  for (int i = 0; i < dimension_; i++) {
    coeff *= steps[i] / 3.0;
  }

  result_ = coeff * sum;
  return true;
}

bool IntegralsSimpsonSequential::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  out_ptr[0] = result_;
  return true;
}

}  // namespace anufriev_d_integrals_simpson_seq