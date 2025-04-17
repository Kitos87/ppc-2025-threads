#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/tyurin_m_matmul_crs_complex/include/ops_seq.hpp"

namespace {
Matrix RandMatrix(uint32_t rows, uint32_t cols, double percentage) {
  std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<double> distr(-100, 100);
  Matrix res{.rows = rows, .cols = cols, .data = std::vector<std::complex<double>>(rows * cols)};
  std::ranges::generate(res.data, [&]() {
    const auto el = distr(gen);
    const auto re = (el < (distr.min() + ((distr.max() - distr.min()) * percentage))) ? el : 0;

    std::complex<double> cmplx;
    cmplx.real(re);
    if (re != 0.0) {
      cmplx.imag(distr(gen));
    }

    return cmplx;
  });
  return res;
}
}  // namespace

TEST(tyurin_m_matmul_crs_complex_seq, test_pipeline_run) {
  auto lhs = RandMatrix(730, 730, 0.22);
  auto rhs = RandMatrix(730, 730, 0.22);

  MatrixCRS crs_lhs = RegularToCRS(lhs);
  MatrixCRS crs_rhs = RegularToCRS(rhs);
  MatrixCRS crs_out;

  auto data = std::make_shared<ppc::core::TaskData>();
  data->inputs = {reinterpret_cast<uint8_t *>(&crs_lhs), reinterpret_cast<uint8_t *>(&crs_rhs)};
  data->inputs_count = {lhs.rows, lhs.cols, rhs.rows, rhs.cols};
  data->outputs = {reinterpret_cast<uint8_t *>(&crs_out)};
  data->outputs_count = {1};

  auto task = std::make_shared<tyurin_m_matmul_crs_complex_seq::TestTaskSequential>(data);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(tyurin_m_matmul_crs_complex_seq, test_task_run) {
  auto lhs = RandMatrix(730, 730, 0.22);
  auto rhs = RandMatrix(730, 730, 0.22);

  MatrixCRS crs_lhs = RegularToCRS(lhs);
  MatrixCRS crs_rhs = RegularToCRS(rhs);
  MatrixCRS crs_out;

  auto data = std::make_shared<ppc::core::TaskData>();
  data->inputs = {reinterpret_cast<uint8_t *>(&crs_lhs), reinterpret_cast<uint8_t *>(&crs_rhs)};
  data->inputs_count = {lhs.rows, lhs.cols, rhs.rows, rhs.cols};
  data->outputs = {reinterpret_cast<uint8_t *>(&crs_out)};
  data->outputs_count = {1};

  auto task = std::make_shared<tyurin_m_matmul_crs_complex_seq::TestTaskSequential>(data);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}
