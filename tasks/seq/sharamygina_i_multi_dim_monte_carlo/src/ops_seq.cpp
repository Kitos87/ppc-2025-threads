#include "seq/sharamygina_i_multi_dim_monte_carlo/include/ops_seq.h"

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <random>
#include <vector>

bool sharamygina_i_multi_dim_monte_carlo_seq::MultiDimMonteCarloTask::PreProcessingImpl() {
  auto* raw_bounds = reinterpret_cast<double*>(task_data->inputs[0]);
  size_t total_bounds_count = task_data->inputs_count[0];
  boundaries_.resize(total_bounds_count);
  std::copy(raw_bounds, raw_bounds + total_bounds_count, boundaries_.begin());
  auto* function_ptr = reinterpret_cast<std::function<double(const std::vector<double>&)>*>(task_data->inputs[2]);
  integrating_function_ = *function_ptr;
  int* iter_ptr = reinterpret_cast<int*>(task_data->inputs[1]);
  number_of_iterations_ = *iter_ptr;

  return true;
}

bool sharamygina_i_multi_dim_monte_carlo_seq::MultiDimMonteCarloTask::ValidationImpl() {
  return task_data && !task_data->outputs_count.empty() && !task_data->outputs.empty() && !task_data->inputs.empty() &&
         task_data->outputs_count[0] == 1 && (task_data->inputs_count.size() == 3) &&
         (task_data->inputs_count[0] % 2 == 0) && (task_data->inputs_count[1] == 1) &&
         (task_data->inputs_count[2] == 1);
}

bool sharamygina_i_multi_dim_monte_carlo_seq::MultiDimMonteCarloTask::RunImpl() {
  size_t dimension = boundaries_.size() / 2;

  std::mt19937 engine(static_cast<unsigned long>(std::time(nullptr)));
  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  double accumulator = 0.0;

  for (int n = 0; n < number_of_iterations_; ++n) {
    std::vector<double> random_point(dimension);
    for (size_t i = 0; i < dimension; ++i) {
      double low = boundaries_[2 * i];
      double high = boundaries_[(2 * i) + 1];
      double t = distribution(engine);
      random_point[i] = low + (high - low) * t;
    }
    accumulator += integrating_function_(random_point);
  }

  double volume = 1.0;
  for (size_t i = 0; i < dimension; ++i) {
    double edge_length = boundaries_[(2 * i) + 1] - boundaries_[2 * i];
    volume *= edge_length;
  }

  final_result_ = (accumulator / static_cast<double>(number_of_iterations_)) * volume;

  return true;
}

bool sharamygina_i_multi_dim_monte_carlo_seq::MultiDimMonteCarloTask::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  output_ptr[0] = final_result_;
  return true;
}
