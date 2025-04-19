#include <gtest/gtest.h>

#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/muhina_m_dijkstra/include/ops_seq.hpp"

namespace {

std::vector<std::vector<std::pair<size_t, int>>> GenerateLargeGraph(size_t k_num_vertices) {
  std::vector<std::vector<std::pair<size_t, int>>> adj_list(k_num_vertices);
  for (size_t i = 0; i < k_num_vertices; ++i) {
    for (size_t j = 0; j < k_num_vertices; ++j) {
      if (i != j) {
        if (rand() % 3 == 0) {
          int weight = (rand() % 10) + 1;
          adj_list[i].emplace_back(j, weight);
        }
      }
    }
  }
  return adj_list;
}
std::vector<int> ConvertGraphToData(const std::vector<std::vector<std::pair<size_t, int>>>& adj_list) {
  std::vector<int> graph_data;
  for (const auto& vertex_edges : adj_list) {
    for (const auto& edge : vertex_edges) {
      graph_data.push_back(static_cast<int>(edge.first));
      graph_data.push_back(edge.second);
    }
    graph_data.push_back(-1);
  }
  return graph_data;
}
}  // namespace

TEST(muhina_m_dijkstra_seq, test_pipeline_run) {
  constexpr size_t kNumVertices = 6000;
  size_t start_vertex = 0;

  auto adj_list = GenerateLargeGraph(kNumVertices);
  auto graph_data = ConvertGraphToData(adj_list);

  std::vector<int> distances(kNumVertices, INT_MAX);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph_data.data()));
  task_data_seq->inputs_count.emplace_back(graph_data.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&start_vertex));
  task_data_seq->inputs_count.emplace_back(sizeof(start_vertex));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));
  task_data_seq->outputs_count.emplace_back(kNumVertices);

  auto test_task_sequential = std::make_shared<muhina_m_dijkstra_seq::TestTaskSequential>(task_data_seq);

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
}

TEST(muhina_m_dijkstra_seq, test_task_run) {
  constexpr size_t kNumVertices = 6000;
  size_t start_vertex = 0;

  auto adj_list = GenerateLargeGraph(kNumVertices);
  auto graph_data = ConvertGraphToData(adj_list);

  std::vector<int> distances(kNumVertices, INT_MAX);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph_data.data()));
  task_data_seq->inputs_count.emplace_back(graph_data.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&start_vertex));
  task_data_seq->inputs_count.emplace_back(sizeof(start_vertex));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));
  task_data_seq->outputs_count.emplace_back(kNumVertices);

  auto test_task_sequential = std::make_shared<muhina_m_dijkstra_seq::TestTaskSequential>(task_data_seq);

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
}
