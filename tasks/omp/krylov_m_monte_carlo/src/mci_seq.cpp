#include "../include/mci_seq.hpp"

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

bool krylov_m_monte_carlo::TaskSequential::RunImpl() {
  const auto dimensions = params->Dimensions();
  const auto iterations = params->iterations;
  const auto func = params->func;

  std::random_device dev;
  std::mt19937 gen(dev());

  std::vector<double> x(dimensions);
  double sum = 0.;
  for (std::size_t _ = 0; _ < iterations; ++_) {
    for (std::size_t p = 0; p < dimensions; ++p) {
      x[p] = dists[p](gen);
    }
    sum += func(x);
  }

  res = (vol * sum) / static_cast<double>(iterations);

  return true;
}