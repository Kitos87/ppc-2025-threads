#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numbers>
#include <numeric>
#include <utility>
#include <vector>

#include "../include/mci_all.hpp"
#include "../include/mci_common.hpp"
#include "../include/mci_seq.hpp"
#include "core/task/include/task.hpp"

using namespace krylov_m_monte_carlo;

using DeterminedTestCase = std::pair<IntegrationParams, double>;

class krylov_m_monte_carlo_all_test  // NOLINT(readability-identifier-naming)
    : public ::testing::TestWithParam<DeterminedTestCase> {
 protected:
  void RunTest(IntegrationParams&& params, double ref) {  // NOLINT(readability-function-cognitive-complexity)
    double out{};

    IntegrationParams workers_params = {.func = params.func, .bounds = {}, .iterations = {}};
    std::shared_ptr<ppc::core::TaskData> task_data =
        world.rank() == 0 ? params.CreateTaskData(out) : workers_params.CreateTaskData(out);

    TaskALL task(task_data);
    ASSERT_TRUE(task.Validation());
    task.PreProcessing();
    task.Run();
    task.PostProcessing();

    if (world.rank() == 0) {
      const double eps = std::abs(ref - out) / out;
      if (ref == 0 || std::isnan(eps)) {
        EXPECT_NEAR(out, ref, 0.42);
      } else {
        EXPECT_LE(eps, 0.1) << "actual: " << out << ", ref: " << ref;
      }
    }
  }
  void RunUndeterminedTest(IntegrationParams&& params) {
    double out{};

    if (world.rank() == 0) {
      TaskSequential task(params.CreateTaskData(out));
      ASSERT_TRUE(task.Validation());
      task.PreProcessing();
      task.Run();
      task.PostProcessing();
    }

    RunTest(std::move(params), out);
  }

  static IntegrationParams GenerateSampleParams(std::size_t dimensions, MathFunction func, std::size_t iterations) {
    return {.func = func, .bounds = std::vector<Bound>(dimensions, {0., 1.}), .iterations = iterations};
  }

  boost::mpi::communicator world;
};

TEST_F(krylov_m_monte_carlo_all_test, sample_1d) {
  RunUndeterminedTest(
      GenerateSampleParams(1, [](const Point& x) { return std::pow(x[0], 2) + std::sin(x[0]); }, 50'000));
}
TEST_F(krylov_m_monte_carlo_all_test, sample_2d) {
  RunUndeterminedTest(GenerateSampleParams(
      2, [](const Point& x) { return (std::pow(x[0], 3) * std::sin(x[0])) + std::exp(x[1]); }, 50'000));
}
TEST_F(krylov_m_monte_carlo_all_test, sample_3d) {
  RunUndeterminedTest(GenerateSampleParams(
      3, [](const Point& x) { return (std::pow(x[0], 4) * std::sin(x[0])) + (std::exp(x[1]) * std::log(x[2])); },
      50'000));
}
TEST_F(krylov_m_monte_carlo_all_test, sample_4d) {
  RunUndeterminedTest(GenerateSampleParams(
      4,
      [](const Point& x) {
        return (std::pow(x[0], 5) * std::sin(x[0])) + std::exp(x[1]) + (std::log(x[2]) * std::tan(x[3]));
      },
      50'000));
}
TEST_F(krylov_m_monte_carlo_all_test, sample_5d) {
  RunUndeterminedTest(GenerateSampleParams(
      5, [](const Point& x) { return std::pow(std::numbers::e, -std::reduce(x.begin(), x.end(), 0.)); }, 50'000));
}

TEST_F(krylov_m_monte_carlo_all_test, validation_failure) {
  if (world.rank() != 0) {
    return;
  }

  IntegrationParams params{.func = [](const Point&) { return 0.; }, .bounds = {{1, 0}}, .iterations = 80'000};
  double stub{};

  TaskALL task(params.CreateTaskData(stub));
  EXPECT_FALSE(task.Validation());
}

TEST_P(krylov_m_monte_carlo_all_test, determined) {
  auto [params, ref] = GetParam();
  RunTest(std::move(params), ref);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(krylov_m_monte_carlo_all_test, krylov_m_monte_carlo_all_test, ::testing::Values( // NOLINT(misc-use-anonymous-namespace)
    DeterminedTestCase{
        {
            .func = [](const Point& x) {
                return 0.;
            },
            .bounds = {
                {-5, 5}
            },
            .iterations = 82'000
        },
        0.
    },
    DeterminedTestCase{
        {
            .func = [](const Point& x) {
                return 5.;
            },
            .bounds = {
                {-5, 5}
            },
            .iterations = 82'000
        },
        50.
    },
    DeterminedTestCase{
        {
            .func = [](const Point& x) {
                return std::sin(x[0]);
            },
            .bounds = {
                {42, 42}
            },
            .iterations = 82'000
        },
        0.
    },
    DeterminedTestCase{
        {
            .func = [](const Point& x) {
                return ((std::cos(x[0]) + 1) * (-std::sin(x[0]))) - std::cos(2 * x[0]);
            },
            .bounds = {
                {-std::numbers::pi, std::numbers::pi}
            },
            .iterations = 82'000
        },
        0.
    },
    DeterminedTestCase{
        {
            .func = [](const Point& x) {
                return ((std::cos(x[0]) + 1) * (-std::sin(x[0]))) - std::cos(2 * x[0]);
            },
            .bounds = {
                {-std::numbers::pi, std::numbers::pi}
            },
            .iterations = 82'000
        },
        0.
    },
    DeterminedTestCase{
        {
            .func = [](const Point& x) {
                return ((std::sin(2 * x[0])) * (-4 * std::sin(2 * x[0]))) - (0.5 * std::cos(x[0])) - (1.5 * std::cos(3 * x[0]));
            },
            .bounds = {
                {-std::numbers::pi, std::numbers::pi}
            },
            .iterations = 82'000
        },
        -4 * std::numbers::pi
    },
    DeterminedTestCase{
        {
            .func = [](const Point& x) {
                return std::pow(x[0], 2) + (4 * x[1]);
            },
            .bounds = {
                {11, 14},
                {7, 10}
            },
            .iterations = 82'000
        },
        []{
            constexpr auto kAntiderivative = [](const double x) { return std::pow(x, 3) + (102 * x); };
            return kAntiderivative(14) - kAntiderivative(11);
        }()
    },
    DeterminedTestCase{
        {
            .func = [](const Point& x) {
                return 9 * std::pow(x[0], 3) * std::pow(x[1], 2);
            },
            .bounds = {
                {1, 3},
                {2, 4}
            },
            .iterations = 82'000
        },
        []{
            constexpr auto kAntiderivative = [](const double x) { return 42 * std::pow(x, 4); };
            return kAntiderivative(3) - kAntiderivative(1);
        }()
    }
));
// clang-format on
