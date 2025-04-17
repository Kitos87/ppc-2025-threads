#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/kolodkin_g_multiplication_matrix_CRS/include/ops_seq.hpp"

TEST(kolodkin_g_multiplication_matrix__task_seq, test_pipeline_run) {
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS a(400, 400);
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS b(400, 400);
  std::vector<Complex> in = {};
  std::vector<Complex> in_a;
  std::vector<Complex> in_b;
  std::vector<Complex> out(a.numCols * b.numRows * 100, 0);

  for (unsigned int i = 0; i < 150; i++) {
    for (unsigned int j = 0; j < 150; j++) {
      a.AddValue((int)i, Complex(-100 + (rand() % 100), -100 + (rand() % 100)), (int)j);
    }
  }
  for (unsigned int i = 50; i < 140; i++) {
    for (unsigned int j = 50; j < 150; j++) {
      b.AddValue((int)i, Complex(-100 + (rand() % 100), -100 + (rand() % 100)), (int)j);
    }
  }
  in_a = kolodkin_g_multiplication_matrix_seq::ParseMatrixIntoVec(a);
  in_b = kolodkin_g_multiplication_matrix_seq::ParseMatrixIntoVec(b);
  in.reserve(in_a.size() + in_b.size());
  for (unsigned int i = 0; i < in_a.size(); i++) {
    in.emplace_back(in_a[i]);
  }
  for (unsigned int i = 0; i < in_b.size(); i++) {
    in.emplace_back(in_b[i]);
  }

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_sequential = std::make_shared<kolodkin_g_multiplication_matrix_seq::TestTaskSequential>(task_data_seq);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS res =
      kolodkin_g_multiplication_matrix_seq::ParseVectorIntoMatrix(out);
}

TEST(kolodkin_g_multiplication_matrix__task_seq, test_task_run) {
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS a(400, 400);
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS b(400, 400);
  std::vector<Complex> in = {};
  std::vector<Complex> in_a;
  std::vector<Complex> in_b;
  std::vector<Complex> out(a.numCols * b.numRows * 100, 0);

  for (unsigned int i = 0; i < 150; i++) {
    for (unsigned int j = 0; j < 150; j++) {
      a.AddValue((int)i, Complex(-100 + (rand() % 100), -100 + (rand() % 100)), (int)j);
    }
  }
  for (unsigned int i = 50; i < 140; i++) {
    for (unsigned int j = 50; j < 150; j++) {
      b.AddValue((int)i, Complex(-100 + (rand() % 100), -100 + (rand() % 100)), (int)j);
    }
  }
  in_a = kolodkin_g_multiplication_matrix_seq::ParseMatrixIntoVec(a);
  in_b = kolodkin_g_multiplication_matrix_seq::ParseMatrixIntoVec(b);
  in.reserve(in_a.size() + in_b.size());
  for (unsigned int i = 0; i < in_a.size(); i++) {
    in.emplace_back(in_a[i]);
  }
  for (unsigned int i = 0; i < in_b.size(); i++) {
    in.emplace_back(in_b[i]);
  }

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_sequential = std::make_shared<kolodkin_g_multiplication_matrix_seq::TestTaskSequential>(task_data_seq);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  kolodkin_g_multiplication_matrix_seq::SparseMatrixCRS res =
      kolodkin_g_multiplication_matrix_seq::ParseVectorIntoMatrix(out);
}
