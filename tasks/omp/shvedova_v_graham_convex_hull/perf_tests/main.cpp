#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "../include/ops_omp.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

constexpr int kCount = 15000;

namespace {
std::vector<double> GenSrc(int count) {
  std::vector<double> points;
  points.reserve(count * 2);
  std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<> dis(-2.55, 4.05);
  for (int i = 0; i < count; ++i) {
    points.push_back(dis(gen));
    points.push_back(dis(gen));
  }
  return points;
}
}  // namespace

TEST(shvedova_v_graham_convex_hull_omp, test_pipeline_run) {
  std::vector<double> points = GenSrc(kCount);
  int scan_size = 0;
  std::vector<double> hull(points.size(), 0.0);
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(points.data()));
  task_data_omp->inputs_count.emplace_back(points.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(&scan_size));
  task_data_omp->outputs_count.emplace_back(1);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(hull.data()));
  task_data_omp->outputs_count.emplace_back(hull.size());
  auto test_task_ompuential = std::make_shared<shvedova_v_graham_convex_hull_omp::GrahamConvexHullOMP>(task_data_omp);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_ompuential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(shvedova_v_graham_convex_hull_omp, test_task_run) {
  std::vector<double> points = GenSrc(kCount);
  int scan_size = 0;
  std::vector<double> hull(points.size(), 0.0);
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(points.data()));
  task_data_omp->inputs_count.emplace_back(points.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(&scan_size));
  task_data_omp->outputs_count.emplace_back(1);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(hull.data()));
  task_data_omp->outputs_count.emplace_back(hull.size());
  auto test_task_ompuential = std::make_shared<shvedova_v_graham_convex_hull_omp::GrahamConvexHullOMP>(task_data_omp);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_ompuential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}
