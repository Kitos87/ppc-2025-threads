#include "seq/poroshin_v_multi_integral_with_trapez_method/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

void poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential::CountMultiIntegralTrapezMethodSeq() {
  int dimensions = static_cast<int>(limits_.size());
  std::vector<double> h(dimensions);
  for (int i = 0; i < dimensions; ++i) {
    h[i] = (limits_[i].second - limits_[i].first) / n_[i];
  }

  double integral = 0.0;
  std::vector<double> vars(dimensions);

  std::vector<int> indices(dimensions, 0);
  int flag = 1;
  while (flag == 1) {
    for (int i = 0; i < dimensions; ++i) {
      vars[i] = limits_[i].first + indices[i] * h[i];
    }

    double weight = 1.0;
    for (int i = 0; i < dimensions; ++i) {
      weight *= (indices[i] == 0 || indices[i] == n_[i]) ? 0.5 : 1.0;
    }
    integral += func_(vars) * weight;

    int dim = 0;
    while (dim < dimensions) {
      indices[dim]++;
      if (indices[dim] <= n_[dim]) {
        break;
      }
      indices[dim] = 0;
      dim++;
      if (dim == dimensions) {
        flag = 0;
        break;
      }
    }
  }

  double volume = 1.0;
  for (int i = 0; i < dimensions; ++i) {
    volume *= h[i];
  }

  res_ = integral * volume;
}

bool poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential::PreProcessingImpl() {
  n_.resize(dim_);
  limits_.resize(dim_);
  for (size_t i = 0; i < dim_; i++) {
    n_[i] = reinterpret_cast<int *>(task_data->inputs[0])[i];
    limits_[i].first = reinterpret_cast<double *>(task_data->inputs[1])[i];
    limits_[i].second = reinterpret_cast<double *>(task_data->inputs[2])[i];
  }
  res_ = 0;
  return true;
}

bool poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential::ValidationImpl() {
  return (task_data->inputs_count[0] > 0 && task_data->outputs_count[0] == 1 && task_data->inputs_count[0] == dim_);
}

bool poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential::RunImpl() {
  CountMultiIntegralTrapezMethodSeq();
  return true;
}

bool poroshin_v_multi_integral_with_trapez_method_seq::TestTaskSequential::PostProcessingImpl() {
  reinterpret_cast<double *>(task_data->outputs[0])[0] = res_;
  return true;
}