#include "omp/kholin_k_multidimensional_integrals_rectangle/include/ops_omp.hpp"

#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

double kholin_k_multidimensional_integrals_rectangle_omp::TestTaskOpenMP::Integrate(
    const Function& f, const std::vector<double>& l_limits, const std::vector<double>& u_limits,
    const std::vector<double>& h, std::vector<double> f_values, int curr_index_dim, size_t dim, double n) {
  if (curr_index_dim == static_cast<int>(dim)) {
    return f(f_values);
  }

  double sum = 0.0;
#pragma omp parallel for reduction(+ : sum) schedule(dynamic)
  for (int i = 0.0; i < static_cast<int>(n); ++i) {
    f_values[curr_index_dim] = l_limits[curr_index_dim] + (static_cast<double>(i) + 0.5) * h[curr_index_dim];
    sum += Integrate(f, l_limits, u_limits, h, f_values, curr_index_dim + 1, dim, n);
  }
  return sum * h[curr_index_dim];
}

double kholin_k_multidimensional_integrals_rectangle_omp::TestTaskOpenMP::IntegrateWithRectangleMethod(
    const Function& f, std::vector<double>& f_values, const std::vector<double>& l_limits,
    const std::vector<double>& u_limits, size_t dim, double n, std::vector<double> h) {
  for (size_t i = 0; i < dim; ++i) {
    h[i] = (u_limits[i] - l_limits[i]) / n;
  }

  return Integrate(f, l_limits, u_limits, h, f_values, 0, dim, n);
}

double kholin_k_multidimensional_integrals_rectangle_omp::TestTaskOpenMP::RunMultistepSchemeMethodRectangle(
    const Function& f, std::vector<double> f_values, const std::vector<double>& l_limits,
    const std::vector<double>& u_limits, size_t dim, double n) {
  std::vector<double> h(dim);
  double i_n = 0.0;
  i_n = IntegrateWithRectangleMethod(f, f_values, l_limits, u_limits, dim, n, h);
  return i_n;
}

bool kholin_k_multidimensional_integrals_rectangle_omp::TestTaskOpenMP::PreProcessingImpl() {
  // Init value for input and output
  sz_values_ = task_data->inputs_count[0];
  sz_lower_limits_ = task_data->inputs_count[1];
  sz_upper_limits_ = task_data->inputs_count[2];

  auto* ptr_dim = reinterpret_cast<size_t*>(task_data->inputs[0]);
  dim_ = *ptr_dim;

  auto* ptr_f_values = reinterpret_cast<double*>(task_data->inputs[1]);
  f_values_.assign(ptr_f_values, ptr_f_values + sz_values_);

  auto* ptr_f = reinterpret_cast<std::function<double(const std::vector<double>&)>*>(task_data->inputs[2]);
  f_ = *ptr_f;

  auto* ptr_lower_limits = reinterpret_cast<double*>(task_data->inputs[3]);
  lower_limits_.assign(ptr_lower_limits, ptr_lower_limits + sz_lower_limits_);

  auto* ptr_upper_limits = reinterpret_cast<double*>(task_data->inputs[4]);
  upper_limits_.assign(ptr_upper_limits, ptr_upper_limits + sz_upper_limits_);

  auto* ptr_start_n = reinterpret_cast<double*>(task_data->inputs[5]);
  start_n_ = *ptr_start_n;

  result_ = 0.0;
  return true;
}

bool kholin_k_multidimensional_integrals_rectangle_omp::TestTaskOpenMP::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs_count[1] > 0U && task_data->inputs_count[2] > 0U;
}

bool kholin_k_multidimensional_integrals_rectangle_omp::TestTaskOpenMP::RunImpl() {
  result_ = RunMultistepSchemeMethodRectangle(f_, f_values_, lower_limits_, upper_limits_, dim_, start_n_);
  return true;
}

bool kholin_k_multidimensional_integrals_rectangle_omp::TestTaskOpenMP::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = result_;
  return true;
}