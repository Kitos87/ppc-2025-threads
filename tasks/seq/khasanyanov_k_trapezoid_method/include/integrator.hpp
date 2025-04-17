#ifndef INTEGRATOR_HPP
#define INTEGRATOR_HPP
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>
namespace khasanyanov_k_trapezoid_method_seq {

enum IntegrationTechnology : std::uint8_t { kSequential, kOpenMP, kTBB, kSTL, kMPI };

using IntegrationFunction = std::function<double(const std::vector<double>&)>;
using Bounds = std::pair<double, double>;
using IntegrationBounds = std::vector<Bounds>;

template <IntegrationTechnology technology>
class Integrator {
  static const int kDefaultSteps, kMaxSteps;

  [[nodiscard]] static double CalculateWeight(const std::vector<int>& indices, int steps);

  [[nodiscard]] static double TrapezoidalMethod(const IntegrationFunction& f, const IntegrationBounds& bounds,
                                                int steps);

  [[nodiscard]] static double TrapezoidalMethodSequential(const IntegrationFunction&, const IntegrationBounds&, double,
                                                          int, int);

 public:
  double operator()(const IntegrationFunction&, const IntegrationBounds&, double, int = kDefaultSteps,
                    int = kMaxSteps) const;
};

//----------------------------------------------------------------------------------------------------------

template <IntegrationTechnology technology>
const int Integrator<technology>::kDefaultSteps = 10;

template <IntegrationTechnology technology>
const int Integrator<technology>::kMaxSteps = 250;

template <IntegrationTechnology technology>
double Integrator<technology>::operator()(const IntegrationFunction& f, const IntegrationBounds& bounds,
                                          double precision, int init_steps, int max_steps) const {
  switch (technology) {
    case kSequential:
      return TrapezoidalMethodSequential(f, bounds, precision, init_steps, max_steps);
    case kTBB:
    case kMPI:
    case kOpenMP:
    case kSTL:
    default:
      throw std::runtime_error("Technology not available");
  }
}

template <IntegrationTechnology technology>
double Integrator<technology>::TrapezoidalMethodSequential(const IntegrationFunction& f,
                                                           const IntegrationBounds& bounds, double precision,
                                                           int init_steps, int max_steps) {
  int steps = init_steps;
  double prev_result = TrapezoidalMethod(f, bounds, steps);
  while (steps <= max_steps) {
    steps *= 2;
    double current_result = TrapezoidalMethod(f, bounds, steps);
    if (std::abs(current_result - prev_result) < precision) {
      return current_result;
    }
    prev_result = current_result;
  }
  return prev_result;
}

template <IntegrationTechnology technology>
double Integrator<technology>::TrapezoidalMethod(const IntegrationFunction& f, const IntegrationBounds& bounds,
                                                 int steps) {
  size_t dims = bounds.size();
  std::vector<double> dx(dims);

  for (size_t i = 0; i < dims; ++i) {
    if (bounds[i].second < bounds[i].first) {
      throw std::runtime_error("Wrong bounds");
    }
    dx[i] = (bounds[i].second - bounds[i].first) / steps;
  }

  double total = 0.0;
  std::vector<int> indices(dims, 0);
  bool done = false;

  while (!done) {
    std::vector<double> point(dims);
    for (size_t i = 0; i < dims; ++i) {
      point[i] = bounds[i].first + indices[i] * dx[i];
    }

    double weight = CalculateWeight(indices, steps);
    total += weight * f(point);

    size_t j = 0;
    while (j < dims) {
      indices[j]++;
      if (indices[j] <= steps) {
        break;
      }

      indices[j] = 0;
      ++j;
    }
    if (j == dims) {
      done = true;
    }
  }

  double factor = 1.0;
  for (size_t i = 0; i < dims; ++i) {
    factor *= dx[i];
  }
  return total * factor;
}

template <IntegrationTechnology technology>
double Integrator<technology>::CalculateWeight(const std::vector<int>& indices, int steps) {
  double weight = 1.0;
  for (int idx : indices) {
    weight *= (idx == 0 || idx == steps) ? 0.5 : 1.0;
  }
  return weight;
}

}  // namespace khasanyanov_k_trapezoid_method_seq

#endif