#include "../include/mci_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <future>
#include <numeric>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

bool krylov_m_monte_carlo::TaskSTL::RunImpl() {
  const auto dimensions = params->Dimensions();
  const auto iterations = params->iterations;
  const auto func = params->func;

  std::random_device dev;
  std::mt19937 gen(dev());

  const auto calculation_thread = [&](std::size_t local_iterations, std::promise<double> &&promise) {
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

  const std::size_t workers = ppc::util::GetPPCNumThreads();
  std::vector<std::future<double>> futures(workers);
  {
    const std::size_t amount = iterations / workers;
    const std::size_t threshold = iterations % workers;

    std::vector<std::thread> threads(workers);
    for (std::size_t i = 0; i < workers; i++) {
      const std::size_t assigned = amount + ((i < threshold) ? 1 : 0);
      std::promise<double> promise;
      futures[i] = promise.get_future();
      threads[i] = std::thread(calculation_thread, assigned, std::move(promise));
    }
    std::ranges::for_each(threads, [](auto &thread) { thread.join(); });
  }

  const double sum = std::transform_reduce(futures.begin(), futures.end(), 0., std::plus{},
                                           [](std::future<double> &future) { return future.get(); });

  res = (vol * sum) / static_cast<double>(iterations);

  return true;
}