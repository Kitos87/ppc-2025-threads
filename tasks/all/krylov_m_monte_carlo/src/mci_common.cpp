#include "../include/mci_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>

bool krylov_m_monte_carlo::TaskCommon::ValidationImpl() {
  return std::ranges::all_of(IntegrationParams::FromTaskData(*task_data).bounds,
                             [](const Bound& bound) { return bound.second >= bound.first; });
}

bool krylov_m_monte_carlo::TaskCommon::PreProcessingImpl() {
  params = &IntegrationParams::FromTaskData(*task_data);
  ApplyParams();
  return true;
}

bool krylov_m_monte_carlo::TaskCommon::PostProcessingImpl() {
  IntegrationParams::OutputOf(*task_data) = res;
  return true;
}

void krylov_m_monte_carlo::TaskCommon::ApplyParams() {
  res = {};
  vol = 1.;
  //
  const std::size_t dimensions = params->Dimensions();
  dists.resize(dimensions);
  //

  auto dist_it = dists.begin();
  for (const auto& bound : params->bounds) {
    *(dist_it++) = std::uniform_real_distribution<double>{bound.first, bound.second};
    vol *= bound.second - bound.first;
  }
}