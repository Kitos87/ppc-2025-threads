#include <gtest/gtest.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/muhina_m_dijkstra/include/ops_seq.hpp"

TEST(muhina_m_dijkstra_seq, test_dijkstra_small_graph) {
  constexpr size_t kNumVertices = 5;
  std::vector<std::vector<std::pair<size_t, int>>> adj_list(kNumVertices);
  adj_list[0].emplace_back(1, 4);
  adj_list[0].emplace_back(2, 2);
  adj_list[1].emplace_back(2, 5);
  adj_list[1].emplace_back(3, 10);
  adj_list[2].emplace_back(3, 3);
  adj_list[3].emplace_back(4, 4);
  adj_list[2].emplace_back(4, 1);

  size_t start_vertex = 0;

  std::vector<int> distances(kNumVertices, INT_MAX);
  std::vector<int> graph_data;
  for (const auto& vertex_edges : adj_list) {
    for (const auto& edge : vertex_edges) {
      graph_data.push_back(static_cast<int>(edge.first));
      graph_data.push_back(edge.second);
    }
    graph_data.push_back(-1);
  }
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph_data.data()));
  task_data_seq->inputs_count.emplace_back(graph_data.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&start_vertex));
  task_data_seq->inputs_count.emplace_back(sizeof(start_vertex));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));
  task_data_seq->outputs_count.emplace_back(kNumVertices);

  muhina_m_dijkstra_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  ASSERT_TRUE(test_task_sequential.Run());
  test_task_sequential.PostProcessing();

  std::vector<int> expected_distances = {0, 4, 2, 5, 3};
  for (size_t i = 0; i < kNumVertices; ++i) {
    EXPECT_EQ(distances[i], expected_distances[i]);
  }
}

TEST(muhina_m_dijkstra_seq, test_dijkstra_validation_failure) {
  std::vector<int> graph_data;
  size_t start_vertex = 0;
  size_t num_vertices = 0;
  std::vector<int> distances(num_vertices, INT_MAX);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph_data.data()));
  task_data_seq->inputs_count.emplace_back(graph_data.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&start_vertex));
  task_data_seq->inputs_count.emplace_back(sizeof(start_vertex));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));
  task_data_seq->outputs_count.emplace_back(num_vertices);

  muhina_m_dijkstra_seq::TestTaskSequential test_task_sequential(task_data_seq);

  ASSERT_FALSE(test_task_sequential.Validation());
}

TEST(muhina_m_dijkstra_seq, test_dijkstra_small_graph_non_zero_start) {
  constexpr size_t kNumVertices = 5;
  std::vector<std::vector<std::pair<size_t, int>>> adj_list(kNumVertices);
  adj_list[0].emplace_back(1, 4);
  adj_list[0].emplace_back(2, 2);
  adj_list[1].emplace_back(2, 5);
  adj_list[1].emplace_back(3, 10);
  adj_list[2].emplace_back(3, 3);
  adj_list[3].emplace_back(4, 4);
  adj_list[2].emplace_back(4, 1);
  adj_list[2].emplace_back(0, 2);
  adj_list[2].emplace_back(1, 5);

  size_t start_vertex = 2;

  std::vector<int> distances(kNumVertices, INT_MAX);
  std::vector<int> graph_data;
  for (const auto& vertex_edges : adj_list) {
    for (const auto& edge : vertex_edges) {
      graph_data.push_back(static_cast<int>(edge.first));
      graph_data.push_back(edge.second);
    }
    graph_data.push_back(-1);
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph_data.data()));
  task_data_seq->inputs_count.emplace_back(graph_data.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&start_vertex));
  task_data_seq->inputs_count.emplace_back(sizeof(start_vertex));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));
  task_data_seq->outputs_count.emplace_back(kNumVertices);

  muhina_m_dijkstra_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  ASSERT_TRUE(test_task_sequential.Run());
  test_task_sequential.PostProcessing();

  std::vector<int> expected_distances = {2, 5, 0, 3, 1};
  for (size_t i = 0; i < kNumVertices; ++i) {
    EXPECT_EQ(distances[i], expected_distances[i]);
  }
}

TEST(muhina_m_dijkstra_seq, test_negative_weight) {
  constexpr size_t kNumVertices = 5;
  std::vector<std::vector<std::pair<size_t, int>>> adj_list(kNumVertices);
  adj_list[0].emplace_back(1, -4);
  adj_list[0].emplace_back(2, 2);
  adj_list[1].emplace_back(2, 5);
  adj_list[1].emplace_back(3, 10);
  adj_list[2].emplace_back(3, -3);
  adj_list[3].emplace_back(4, 4);
  adj_list[2].emplace_back(4, 1);

  size_t start_vertex = 0;

  std::vector<int> distances(kNumVertices, INT_MAX);
  std::vector<int> graph_data;
  for (const auto& vertex_edges : adj_list) {
    for (const auto& edge : vertex_edges) {
      graph_data.push_back(static_cast<int>(edge.first));
      graph_data.push_back(edge.second);
    }
    graph_data.push_back(-1);
  }
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(graph_data.data()));
  task_data_seq->inputs_count.emplace_back(graph_data.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&start_vertex));
  task_data_seq->inputs_count.emplace_back(sizeof(start_vertex));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(distances.data()));
  task_data_seq->outputs_count.emplace_back(kNumVertices);

  muhina_m_dijkstra_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  ASSERT_FALSE(test_task_sequential.Run());
  test_task_sequential.PostProcessing();
}
