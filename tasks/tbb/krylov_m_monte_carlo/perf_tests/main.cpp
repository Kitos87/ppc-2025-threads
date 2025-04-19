#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <vector>

#include "../include/mci_common.hpp"
#include "../include/mci_tbb.hpp"
#include "core/perf/include/perf.hpp"

using namespace krylov_m_monte_carlo;

class krylov_m_monte_carlo_test_tbb : public ::testing::Test {  // NOLINT(readability-identifier-naming)
 protected:
  static void RunPerfTest(const std::function<void(ppc::core::Perf &, const std::shared_ptr<ppc::core::PerfAttr> &,
                                                   const std::shared_ptr<ppc::core::PerfResults> &)> &runner) {
    double out{};
    IntegrationParams params{.func =
                                 [](const Point &x) {
                                   return std::log(x[0]) + std::sin(x[1]) + std::cos(x[2]) + std::tan(x[3]) +
                                          std::exp(x[4]);
                                 },
                             .bounds = std::vector<Bound>(5, {0., 1.}),
                             .iterations = 4'000'000};

    auto task = std::make_shared<TaskTBB>(params.CreateTaskData(out));

    //
    auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
    perf_attr->num_running = 10;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perf_attr->current_timer = [&] {
      auto current_time_point = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
      return static_cast<double>(duration) * 1e-9;
    };

    //
    auto perf_results = std::make_shared<ppc::core::PerfResults>();
    ppc::core::Perf perf_analyzer(task);
    runner(perf_analyzer, perf_attr, perf_results);
    ppc::core::Perf::PrintPerfStatistic(perf_results);

    const double ref = 2.634;
    const double eps = std::abs(ref - out) / out;
    EXPECT_LE(eps, 0.1);
  }
};

TEST_F(krylov_m_monte_carlo_test_tbb, test_pipeline_run) {
  RunPerfTest([](auto &perf_analyzer, const auto &perf_attr, const auto &perf_results) {
    perf_analyzer.PipelineRun(perf_attr, perf_results);
  });
}

TEST_F(krylov_m_monte_carlo_test_tbb, test_task_run) {
  RunPerfTest([](auto &perf_analyzer, const auto &perf_attr, const auto &perf_results) {
    perf_analyzer.TaskRun(perf_attr, perf_results);
  });
}
