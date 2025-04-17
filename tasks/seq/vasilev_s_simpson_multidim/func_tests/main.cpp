#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "../include/ops_seq.hpp"
#include "core/task/include/task.hpp"

namespace vasilev_s_simpson_multidim_test {

struct IntegrationTest {
  std::size_t approxs;
  vasilev_s_simpson_multidim::IntegrandFunction ifun;
  std::vector<vasilev_s_simpson_multidim::Bound> bounds;
  double ref;
};

}  // namespace vasilev_s_simpson_multidim_test

class PresetTests : public ::testing::TestWithParam<vasilev_s_simpson_multidim_test::IntegrationTest> {};

TEST_P(PresetTests, run_and_verify) {
  auto test = GetParam();
  double out{};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t *>(test.bounds.data()), reinterpret_cast<uint8_t *>(test.ifun),
                       reinterpret_cast<uint8_t *>(&test.approxs)};
  task_data->inputs_count.emplace_back(test.bounds.size());
  task_data->outputs = {reinterpret_cast<uint8_t *>(&out)};
  task_data->outputs_count.emplace_back(1);

  vasilev_s_simpson_multidim::SimpsonTaskOmp task(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  EXPECT_NEAR(out, test.ref, std::min(0.5, test.ref / 3));
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(vasilev_s_simpson_multidim_test_seq, PresetTests, ::testing::Values( // NOLINT
    vasilev_s_simpson_multidim_test::IntegrationTest{
      .approxs = 32,
      .ifun = [](const auto &coord) { return std::sin(coord[0]); },
      .bounds = {
        {0.0, 1.0},
      },
      .ref = 1 - std::cos(1),
    },
    vasilev_s_simpson_multidim_test::IntegrationTest{
      .approxs = 32,
      .ifun = [](const auto &coord) { return std::sin(coord[0]); },
      .bounds = {
        {0.0, 1.0},
        {0.0, 1.0},
      },
      .ref = 1 - std::cos(1),
    },
    vasilev_s_simpson_multidim_test::IntegrationTest{
      .approxs = 32,
      .ifun = [](const auto &coord) { return std::sin(coord[0]); },
      .bounds = {
        {0.0, 1.0},
        {0.0, 1.0},
        {0.0, 1.0},
      },
      .ref = 1 - std::cos(1),
    },
    vasilev_s_simpson_multidim_test::IntegrationTest{
      .approxs = 32,
      .ifun = [](const auto &coord) { return std::cos(coord[0]); },
      .bounds = {
        {0.0, 1.0},
      },
      .ref = std::sin(1),
    },
    vasilev_s_simpson_multidim_test::IntegrationTest{
      .approxs = 32,
      .ifun = [](const auto &coord) { return std::cos(coord[0]); },
      .bounds = {
        {0.0, 1.0},
        {0.0, 1.0},
      },
      .ref = std::sin(1),
    },
    vasilev_s_simpson_multidim_test::IntegrationTest{
      .approxs = 32,
      .ifun = [](const auto &coord) { return std::cos(coord[0]); },
      .bounds = {
        {0.0, 1.0},
        {0.0, 1.0},
        {0.0, 1.0},
      },
      .ref = std::sin(1),
    },
    vasilev_s_simpson_multidim_test::IntegrationTest{
      .approxs = 32,
      .ifun = [](const auto &coord) { return std::cos(coord[0]) * std::sin(coord[1]) * std::exp(coord[2]); },
      .bounds = {
        {0.0, 1.0},
        {0.0, 1.0},
        {0.0, 1.0},
      },
      .ref = 0.56,
    }
));
// clang-format on
