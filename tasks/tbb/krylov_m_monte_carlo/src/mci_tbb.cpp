#include "../include/mci_tbb.hpp"

#include <oneapi/tbb/parallel_reduce.h>
#include <oneapi/tbb/task_arena.h>
#include <tbb/tbb.h>

#include <cmath>
#include <cstddef>
#include <functional>
#include <random>
#include <vector>

#include "core/util/include/util.hpp"

bool krylov_m_monte_carlo::TaskTBB::RunImpl() {
  const auto dimensions = params->Dimensions();
  const auto iterations = params->iterations;
  const auto func = params->func;

  std::random_device dev;
  std::mt19937 gen(dev());

  oneapi::tbb::task_arena arena(ppc::util::GetPPCNumThreads());
  const double sum = arena.execute([&] {
    return oneapi::tbb::parallel_reduce(
        oneapi::tbb::blocked_range<std::size_t>(0, iterations,
                                                iterations / oneapi::tbb::this_task_arena::max_concurrency()),
        0.0,
        [&](const tbb::blocked_range<std::size_t>& r, double partial_sum) {
          auto local_gen = gen;
          std::vector<double> x(dimensions);
          for (std::size_t _ = r.begin(); _ < r.end(); ++_) {
            for (std::size_t p = 0; p < dimensions; ++p) {
              x[p] = dists[p](local_gen);
            }
            partial_sum += func(x);
          }
          return partial_sum;
        },
        std::plus<>());
  });

  res = (vol * sum) / static_cast<double>(iterations);

  return true;
}