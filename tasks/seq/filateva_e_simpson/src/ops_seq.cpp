#include "seq/filateva_e_simpson/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

bool filateva_e_simpson_seq::Simpson::PreProcessingImpl() {
  mer_ = task_data->inputs_count[0];
  steps_ = task_data->inputs_count[1];

  auto *temp_a = reinterpret_cast<double *>(task_data->inputs[0]);
  a_.insert(a_.end(), temp_a, temp_a + mer_);

  auto *temp_b = reinterpret_cast<double *>(task_data->inputs[1]);
  b_.insert(b_.end(), temp_b, temp_b + mer_);

  f_ = reinterpret_cast<Func>(task_data->inputs[2]);

  return true;
}

bool filateva_e_simpson_seq::Simpson::ValidationImpl() {
  size_t mer = task_data->inputs_count[0];
  auto *temp_a = reinterpret_cast<double *>(task_data->inputs[0]);
  auto *temp_b = reinterpret_cast<double *>(task_data->inputs[1]);
  if (task_data->inputs_count[1] % 2 == 1) {
    return false;
  }
  for (size_t i = 0; i < mer; i++) {
    if (temp_b[i] <= temp_a[i]) {
      return false;
    }
  }
  return true;
}

bool filateva_e_simpson_seq::Simpson::RunImpl() {
  std::vector<double> h(mer_);
  for (size_t i = 0; i < mer_; i++) {
    h[i] = static_cast<double>(b_[i] - a_[i]) / static_cast<double>(steps_);
  }

  res_ = 0.0;

  for (unsigned long i = 0; i < static_cast<unsigned long>(std::pow(steps_ + 1, mer_)); i++) {
    unsigned long temp = i;
    std::vector<double> param(mer_);
    double weight = 1.0;

    for (size_t m = 0; m < mer_; m++) {
      size_t shag_i = temp % (steps_ + 1);
      temp /= (steps_ + 1);

      param[m] = a_[m] + h[m] * static_cast<double>(shag_i);

      if (shag_i == 0 || shag_i == steps_) {
        weight *= 1.0;
      } else if (shag_i % 2 == 1) {
        weight *= 4.0;
      } else {
        weight *= 2.0;
      }
    }

    res_ += weight * f_(param);
  }

  for (size_t i = 0; i < mer_; i++) {
    res_ *= (h[i] / 3.0);
  }
  return true;
}

bool filateva_e_simpson_seq::Simpson::PostProcessingImpl() {
  reinterpret_cast<double *>(task_data->outputs[0])[0] = res_;
  return true;
}
