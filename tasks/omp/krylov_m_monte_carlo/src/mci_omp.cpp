#include "../include/mci_omp.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "../include/mci_common.hpp"

bool krylov_m_monte_carlo::TaskOpenMP::ValidationImpl() {
  return IntegrationParams::FromTaskData(*task_data).iterations <=
             static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) &&
         TaskCommon::ValidationImpl();
}

bool krylov_m_monte_carlo::TaskOpenMP::RunImpl() {
  const auto dimensions = params->Dimensions();
  const auto iterations = static_cast<std::int64_t>(params->iterations);
  const auto func = params->func;

  std::random_device dev;
  std::mt19937 gen(dev());

  double sum = 0.;
#pragma omp parallel firstprivate(gen)
  {
    std::vector<double> x(dimensions);
#pragma omp for reduction(+ : sum)
    for (std::int64_t _ = 0; _ < iterations; ++_) {
      for (std::size_t p = 0; p < dimensions; ++p) {
        x[p] = dists[p](gen);
      }
      sum += func(x);
    }
  }

  res = (vol * sum) / static_cast<double>(iterations);

  return true;
}