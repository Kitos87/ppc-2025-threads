#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <utility>

#include "../include/mci_common.hpp"
#include "../include/mci_seq.hpp"

using namespace krylov_m_monte_carlo;

using DeterminedTestCase = std::pair<IntegrationParams, double>;

class krylov_m_monte_carlo_test_seq  // NOLINT(readability-identifier-naming)
    : public ::testing::TestWithParam<DeterminedTestCase> {
 protected:
  static void RunTest(IntegrationParams& params, double ref) {
    double out{};

    TaskSequential task(params.CreateTaskData(out));
    ASSERT_TRUE(task.Validation());
    task.PreProcessing();
    task.Run();
    task.PostProcessing();

    const double eps = std::abs(ref - out) / out;
    if (ref == 0 || std::isnan(eps)) {
      EXPECT_NEAR(out, ref, 0.42);
    } else {
      EXPECT_LE(eps, 0.1) << "actual: " << out << ", ref: " << ref;
    }
  }
};

TEST_F(krylov_m_monte_carlo_test_seq, validation_failure) {
  IntegrationParams params{.func = [](const Point&) { return 0.; }, .bounds = {{1, 0}}, .iterations = 100'000};
  double stub{};

  TaskSequential task(params.CreateTaskData(stub));
  EXPECT_FALSE(task.Validation());
}

TEST_P(krylov_m_monte_carlo_test_seq, determined) {
  auto [params, ref] = GetParam();
  RunTest(params, ref);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(krylov_m_monte_carlo_test_seq, krylov_m_monte_carlo_test_seq, ::testing::Values( // NOLINT(misc-use-anonymous-namespace)
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
