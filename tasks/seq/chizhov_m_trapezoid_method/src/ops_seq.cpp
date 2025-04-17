#include "seq/chizhov_m_trapezoid_method/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

double chizhov_m_trapezoid_method_seq::TrapezoidMethod(Function& f, size_t div, size_t dim,
                                                       std::vector<double>& lower_limits,
                                                       std::vector<double>& upper_limits) {
  int int_div = static_cast<int>(div);
  int int_dim = static_cast<int>(dim);
  std::vector<double> h(int_dim);
  std::vector<int> steps(int_dim);

  for (int i = 0; i < int_dim; i++) {
    steps[i] = int_div;
    h[i] = (upper_limits[i] - lower_limits[i]) / steps[i];
  }

  int total_nodes = 1;
  for (const auto& step : steps) {
    total_nodes *= (step + 1);
  }

  double result = 0.0;
  std::vector<double> point(int_dim);

  for (int i = 0; i < total_nodes; i++) {
    int temp = i;

    for (int j = 0; j < int_dim; j++) {
      int node_index = temp % (steps[j] + 1);
      point[j] = lower_limits[j] + node_index * h[j];
      temp /= (steps[j] + 1);
    }

    double weight = 1.0;
    for (int j = 0; j < int_dim; j++) {
      if (point[j] == lower_limits[j] || point[j] == upper_limits[j]) {
        weight *= 1.0;
      } else {
        weight *= 2.0;
      }
    }

    result += weight * f(point);
  }

  for (int i = 0; i < int_dim; i++) {
    result *= h[i] / 2.0;
  }

  return std::round(result * 100.0) / 100.0;
}

bool chizhov_m_trapezoid_method_seq::TestTaskSequential::PreProcessingImpl() {
  int* divisions_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  div_ = *divisions_ptr;

  int* dimension_ptr = reinterpret_cast<int*>(task_data->inputs[1]);
  dim_ = *dimension_ptr;

  auto* limit_ptr = reinterpret_cast<double*>(task_data->inputs[2]);
  for (int i = 0; i < static_cast<int>(task_data->inputs_count[2]); i += 2) {
    lower_limits_.push_back(limit_ptr[i]);
    upper_limits_.push_back(limit_ptr[i + 1]);
  }
  auto* ptr_f = reinterpret_cast<std::function<double(const std::vector<double>&)>*>(task_data->inputs[3]);
  f_ = *ptr_f;

  return true;
}

bool chizhov_m_trapezoid_method_seq::TestTaskSequential::ValidationImpl() {
  int* divisions_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  int* dimension_ptr = reinterpret_cast<int*>(task_data->inputs[1]);
  if (*divisions_ptr <= 0 || *dimension_ptr <= 0) {
    return false;
  }
  if (task_data->inputs_count[2] % 2 != 0) {
    return false;
  }
  auto* limit_ptr = reinterpret_cast<double*>(task_data->inputs[2]);
  for (int i = 0; i < static_cast<int>(task_data->inputs_count[2]); i += 2) {
    if (limit_ptr[i] >= limit_ptr[i + 1]) {
      return false;
    }
  }

  return true;
}

bool chizhov_m_trapezoid_method_seq::TestTaskSequential::RunImpl() {
  res_ = TrapezoidMethod(f_, div_, dim_, lower_limits_, upper_limits_);

  return true;
}

bool chizhov_m_trapezoid_method_seq::TestTaskSequential::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = res_;

  return true;
}