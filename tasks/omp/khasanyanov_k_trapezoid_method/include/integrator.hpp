#ifndef INTEGRATOR_HPP
#define INTEGRATOR_HPP

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>
namespace khasanyanov_k_trapezoid_method_omp {

enum IntegrationTechnology : std::uint8_t { kSequential, kOpenMP, kTBB, kSTL, kMPI };

using IntegrationFunction = std::function<double(const std::vector<double>&)>;
using Bounds = std::pair<double, double>;
using IntegrationBounds = std::vector<Bounds>;

template <IntegrationTechnology technology>
class Integrator {
  static const int kDefaultSteps, kMaxSteps;

  [[nodiscard]] static double TrapezoidalMethodSequential(const IntegrationFunction& f, const IntegrationBounds& bounds,
                                                          int steps);

  [[nodiscard]] static double TrapezoidalMethodOmp(const IntegrationFunction& f, const IntegrationBounds& bounds,
                                                   int steps);

  [[nodiscard]] static double TrapezoidalMethod(const IntegrationFunction&, const IntegrationBounds&, double, int, int,
                                                auto ref_func);

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
      return TrapezoidalMethod(f, bounds, precision, init_steps, max_steps, &TrapezoidalMethodSequential);
    case kTBB:
    case kMPI:
    case kOpenMP:
      return TrapezoidalMethod(f, bounds, precision, init_steps, max_steps, &TrapezoidalMethodOmp);
    case kSTL:
    default:
      throw std::runtime_error("Technology not available");
  }
}

template <IntegrationTechnology technology>
double Integrator<technology>::TrapezoidalMethod(const IntegrationFunction& f, const IntegrationBounds& bounds,
                                                 double precision, int init_steps, int max_steps, auto ref_func) {
  int steps = init_steps;
  double prev_result = ref_func(f, bounds, steps);
  while (steps <= max_steps) {
    steps *= 2;
    double current_result = ref_func(f, bounds, steps);
    if (std::abs(current_result - prev_result) < precision) {
      return current_result;
    }
    prev_result = current_result;
  }
  return prev_result;
}

template <IntegrationTechnology technology>
double Integrator<technology>::TrapezoidalMethodSequential(const IntegrationFunction& f,
                                                           const IntegrationBounds& bounds, int steps) {
  const size_t dimension = bounds.size();

  std::vector<double> h(dimension);
  double cell_volume = 1.0;
  for (size_t i = 0; i < dimension; ++i) {
    const auto& [a, b] = bounds[i];
    if (b < a) {
      throw std::runtime_error("Wrong bounds");
    }
    h[i] = (b - a) / steps;
    cell_volume *= h[i];
  }

  int total_points = 1;
  for (size_t i = 0; i < dimension; ++i) {
    total_points *= (steps + 1);
  }

  double total_sum = 0.0;

  for (int idx = 0; idx < total_points; ++idx) {
    std::vector<double> point(dimension);
    int temp = idx;
    int boundary_count = 0;

    for (size_t dim = 0; dim < dimension; ++dim) {
      const int steps_per_dim = steps + 1;
      const int step = temp % steps_per_dim;
      temp /= steps_per_dim;

      const auto& [a, _] = bounds[dim];
      point[dim] = a + step * h[dim];

      if (step == 0 || step == steps) {
        boundary_count++;
      }
    }

    const double weight = std::pow(0.5, boundary_count);
    total_sum += f(point) * weight;
  }

  return total_sum * cell_volume;
}

template <IntegrationTechnology technology>
double Integrator<technology>::TrapezoidalMethodOmp(const IntegrationFunction& f, const IntegrationBounds& bounds,
                                                    int steps) {
  const size_t dimension = bounds.size();

  std::vector<double> h(dimension);
  double cell_volume = 1.0;
  for (size_t i = 0; i < dimension; ++i) {
    const auto& [a, b] = bounds[i];
    if (b < a) {
      throw std::runtime_error("Wrong bounds");
    }
    h[i] = (b - a) / steps;
    cell_volume *= h[i];
  }

  int total_points = 1;
  for (size_t i = 0; i < dimension; ++i) {
    total_points *= (steps + 1);
  }

  double total_sum = 0.0;

#pragma omp parallel for reduction(+ : total_sum)
  for (int idx = 0; idx < total_points; ++idx) {
    std::vector<double> point(dimension);
    int temp = idx;
    int boundary_count = 0;

    for (size_t dim = 0; dim < dimension; ++dim) {
      const int steps_per_dim = steps + 1;
      const int step = temp % steps_per_dim;
      temp /= steps_per_dim;

      const auto& [a, _] = bounds[dim];
      point[dim] = a + step * h[dim];

      if (step == 0 || step == steps) {
        boundary_count++;
      }
    }

    const double weight = std::pow(0.5, boundary_count);
    total_sum += f(point) * weight;
  }

  return total_sum * cell_volume;
}

}  // namespace khasanyanov_k_trapezoid_method_omp

#endif