#include "omp/lopatin_i_monte_carlo/include/lopatinMonteCarloOMP.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ctime>
#include <random>
#include <vector>

namespace lopatin_i_monte_carlo_omp {

bool TestTaskOMP::ValidationImpl() {
  const bool outputs_valid = !task_data->outputs_count.empty() && task_data->outputs_count[0] == 1;
  const bool inputs_valid = task_data->inputs_count.size() == 2 &&
                            (task_data->inputs_count[0] % 2 == 0) &&  // odd num of bounds
                            task_data->inputs_count[1] == 1;          // iterations num

  auto* iter_ptr = reinterpret_cast<int*>(task_data->inputs[1]);
  const int iterations = *iter_ptr;
  const bool iter_valid = iterations > 0;

  return outputs_valid && inputs_valid && iter_valid;
}

bool TestTaskOMP::PreProcessingImpl() {
  auto* bounds_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  size_t bounds_size = task_data->inputs_count[0];
  integrationBounds_.resize(bounds_size);
  std::copy(bounds_ptr, bounds_ptr + bounds_size, integrationBounds_.begin());

  auto* iter_ptr = reinterpret_cast<int*>(task_data->inputs[1]);
  iterations_ = *iter_ptr;
  return true;
}

bool TestTaskOMP::RunImpl() {
  const size_t d = integrationBounds_.size() / 2;  // dimensions

  // init random numbers generator
  std::random_device rd;
  std::seed_seq seed{rd(), static_cast<unsigned int>(std::time(nullptr))};
  std::vector<std::mt19937::result_type> seeds(omp_get_max_threads());
  seed.generate(seeds.begin(), seeds.end());

  // volume of integration region
  double volume = 1.0;
  for (size_t j = 0; j < d; ++j) {
    volume *= (integrationBounds_[(2 * j) + 1] - integrationBounds_[2 * j]);
  }

  double total_sum = 0.0;
#pragma omp parallel reduction(+ : total_sum)
  {
    // init generator for each thread with unique seed
    const int tid = omp_get_thread_num();
    std::mt19937 local_rnd(seeds[tid]);
    std::uniform_real_distribution<> dis(0.0, 1.0);

#pragma omp for
    for (int i = 0; i < iterations_; ++i) {
      std::vector<double> point(d);
      for (size_t j = 0; j < d; ++j) {
        const double min = integrationBounds_[2 * j];
        const double max = integrationBounds_[(2 * j) + 1];
        point[j] = min + (max - min) * dis(local_rnd);
      }
      total_sum += integrand_(point);
    }
  }

  result_ = (total_sum / iterations_) * volume;

  return true;
}

bool TestTaskOMP::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  *output_ptr = result_;
  return true;
}

}  // namespace lopatin_i_monte_carlo_omp
