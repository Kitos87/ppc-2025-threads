#include "../include/mci_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/serialization/utility.hpp>  // NOLINT(*-include-cleaner)
#include <boost/serialization/vector.hpp>   // NOLINT(*-include-cleaner)
#include <cmath>
#include <cstddef>
#include <functional>
#include <future>
#include <numeric>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include "../include/mci_common.hpp"
#include "core/util/include/util.hpp"

bool krylov_m_monte_carlo::TaskALL::ValidationImpl() { return world_.rank() != 0 || TaskCommon::ValidationImpl(); }

bool krylov_m_monte_carlo::TaskALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    return TaskCommon::PreProcessingImpl();
  }

  // just for func
  params = &IntegrationParams::FromTaskData(*task_data);
  return true;
}

bool krylov_m_monte_carlo::TaskALL::PostProcessingImpl() {
  return world_.rank() != 0 || TaskCommon::PostProcessingImpl();
}

bool krylov_m_monte_carlo::TaskALL::RunImpl() {
  const auto func = params->func;
  boost::mpi::broadcast(world_, params, 0);
  if (world_.rank() != 0) {
    params->func = func;
    ApplyParams();
  }

  const auto dimensions = params->Dimensions();

  const auto global_iterations = params->iterations;
  const auto node_iterations = [this, &global_iterations] {
    const auto nodes = world_.size();
    const std::size_t amount = global_iterations / nodes;
    const std::size_t threshold = global_iterations % nodes;
    //
    return amount + (static_cast<std::size_t>(world_.rank()) < threshold ? 1 : 0);
  }();

  std::random_device dev;
  std::mt19937 gen(dev());

  const auto calculation_thread = [&](std::size_t local_iterations, std::promise<double>&& promise) {
    auto local_gen = gen;
    std::vector<double> x(dimensions);
    double partial_sum = 0.;
    for (std::size_t _ = 0; _ < local_iterations; ++_) {
      for (std::size_t p = 0; p < dimensions; ++p) {
        x[p] = dists[p](local_gen);
      }
      partial_sum += func(x);
    }

    promise.set_value(partial_sum);
  };

  const std::size_t node_workers = ppc::util::GetPPCNumThreads();
  std::vector<std::future<double>> futures(node_workers);
  {
    const std::size_t amount = node_iterations / node_workers;
    const std::size_t threshold = node_iterations % node_workers;

    std::vector<std::thread> threads(node_workers);
    for (std::size_t i = 0; i < node_workers; i++) {
      const std::size_t assigned = amount + ((i < threshold) ? 1 : 0);
      std::promise<double> promise;
      futures[i] = promise.get_future();
      threads[i] = std::thread(calculation_thread, assigned, std::move(promise));
    }
    std::ranges::for_each(threads, [](auto& thread) { thread.join(); });
  }

  const double partial_sum = std::transform_reduce(futures.begin(), futures.end(), 0., std::plus{},
                                                   [](std::future<double>& future) { return future.get(); });

  double sum{};
  boost::mpi::reduce(world_, partial_sum, sum, std::plus{}, 0);

  res = (vol * sum) / static_cast<double>(global_iterations);

  return true;
}