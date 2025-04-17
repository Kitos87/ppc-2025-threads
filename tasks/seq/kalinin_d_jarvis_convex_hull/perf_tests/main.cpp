// Copyright 2025 Kalinin Dmitry
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/kalinin_d_jarvis_convex_hull/include/ops_seq.hpp"

namespace {

double Cross(const kalinin_d_jarvis_convex_hull_seq::Point &o, const kalinin_d_jarvis_convex_hull_seq::Point &a,
             const kalinin_d_jarvis_convex_hull_seq::Point &b) {
  return ((a.x - o.x) * (b.y - o.y)) - ((a.y - o.y) * (b.x - o.x));
}

std::vector<kalinin_d_jarvis_convex_hull_seq::Point> CalculateConvexHull(
    std::vector<kalinin_d_jarvis_convex_hull_seq::Point> points) {
  if (points.size() < 3) {
    return points;
  }

  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> hull;

  size_t l = 0;
  for (size_t i = 1; i < points.size(); ++i) {
    if (points[i].x < points[l].x) {
      l = i;
    }
  }

  size_t p = l;
  size_t q = 0;
  do {
    hull.push_back(points[p]);

    q = (p + 1) % points.size();
    for (size_t i = 0; i < points.size(); ++i) {
      if (Cross(points[p], points[i], points[q]) > 0) {
        q = i;
      }
    }

    p = q;

  } while (p != l);

  return hull;
}

}  // namespace

TEST(kalinin_d_jarvis_convex_hull_seq, test_pipeline_run) {
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> points;
  const int size = 10000000;
  points.reserve(size);
  for (int i = 0; i < size; i++) {
    points.push_back({i % 100, i % 200});
  }
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> res_hull(points.size());
  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  task_data_seq->inputs_count.emplace_back(points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_hull.data()));
  task_data_seq->outputs_count.emplace_back(res_hull.size());

  // Create Task
  auto test_task_sequential = std::make_shared<kalinin_d_jarvis_convex_hull_seq::TestTaskSequential>(task_data_seq);

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

  // Verify results
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> res = CalculateConvexHull(points);
  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(res[i], res_hull[i]);
  }
}

TEST(kalinin_d_jarvis_convex_hull_seq, test_task_run) {
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> points;
  const int size = 10000000;
  points.reserve(size);
  for (int i = 0; i < size; i++) {
    points.push_back({rand() % 100, rand() % 200});
  }

  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> res_hull(points.size());
  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(points.data()));
  task_data_seq->inputs_count.emplace_back(points.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_hull.data()));
  task_data_seq->outputs_count.emplace_back(res_hull.size());

  // Create Task
  auto test_task_sequential = std::make_shared<kalinin_d_jarvis_convex_hull_seq::TestTaskSequential>(task_data_seq);

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

  // Verify results
  std::vector<kalinin_d_jarvis_convex_hull_seq::Point> res = CalculateConvexHull(points);
  for (size_t i = 0; i < res.size(); ++i) {
    ASSERT_EQ(res[i], res_hull[i]);
  }
}