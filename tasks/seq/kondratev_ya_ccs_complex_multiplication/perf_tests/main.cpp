#include <gtest/gtest.h>

#include <chrono>
#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/kondratev_ya_ccs_complex_multiplication/include/ops_seq.hpp"

namespace {
void FillMatrix(kondratev_ya_ccs_complex_multiplication_seq::CCSMatrix &matrix, std::complex<double> value) {
  for (int i = 0; i < matrix.rows; i++) {
    matrix.values.emplace_back(value);
    matrix.row_index.emplace_back(i);
    matrix.col_ptrs[i] = i;
  }
  matrix.col_ptrs[matrix.rows] = matrix.rows;
}

void CheckColumnPointers(const kondratev_ya_ccs_complex_multiplication_seq::CCSMatrix &matrix, int count) {
  for (int i = 0; i < count; i++) {
    ASSERT_EQ(matrix.col_ptrs[i], i);
  }
  ASSERT_EQ(matrix.col_ptrs[count], count);
}

void CheckRowIndices(const kondratev_ya_ccs_complex_multiplication_seq::CCSMatrix &matrix, int count) {
  for (int i = 0; i < count; i++) {
    ASSERT_EQ(matrix.row_index[i], i);
  }
}

void CheckValues(const kondratev_ya_ccs_complex_multiplication_seq::CCSMatrix &matrix, int count,
                 const std::complex<double> &value) {
  for (int i = 0; i < count; i++) {
    ASSERT_DOUBLE_EQ(matrix.values[i].real(), value.real());
    ASSERT_DOUBLE_EQ(matrix.values[i].imag(), value.imag());
  }
}

void CheckResult(kondratev_ya_ccs_complex_multiplication_seq::CCSMatrix &matrix, int count,
                 std::complex<double> value) {
  CheckColumnPointers(matrix, count);
  CheckRowIndices(matrix, count);
  CheckValues(matrix, count, value);
}
}  // namespace

TEST(kondratev_ya_ccs_complex_multiplication_seq, test_pipeline_run) {
  constexpr int kCount = 27000;

  kondratev_ya_ccs_complex_multiplication_seq::CCSMatrix a({kCount, kCount});
  kondratev_ya_ccs_complex_multiplication_seq::CCSMatrix b({kCount, kCount});
  kondratev_ya_ccs_complex_multiplication_seq::CCSMatrix c({kCount, kCount});

  FillMatrix(a, {2.0, 1.0});
  FillMatrix(b, {3.0, 2.0});

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  task_data_seq->inputs_count.emplace_back(2);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&c));
  task_data_seq->outputs_count.emplace_back(1);

  auto test_task_sequential =
      std::make_shared<kondratev_ya_ccs_complex_multiplication_seq::TestTaskSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  CheckResult(c, kCount, {4.0, 7.0});
}

TEST(kondratev_ya_ccs_complex_multiplication_seq, test_task_run) {
  constexpr int kCount = 27000;

  kondratev_ya_ccs_complex_multiplication_seq::CCSMatrix a({kCount, kCount});
  kondratev_ya_ccs_complex_multiplication_seq::CCSMatrix b({kCount, kCount});
  kondratev_ya_ccs_complex_multiplication_seq::CCSMatrix c({kCount, kCount});

  FillMatrix(a, {2.0, 1.0});
  FillMatrix(b, {3.0, 2.0});

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&a));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&b));
  task_data_seq->inputs_count.emplace_back(2);

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&c));
  task_data_seq->outputs_count.emplace_back(1);

  auto test_task_sequential =
      std::make_shared<kondratev_ya_ccs_complex_multiplication_seq::TestTaskSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  CheckResult(c, kCount, {4.0, 7.0});
}
