#include "seq/muhina_m_dijkstra/include/ops_seq.hpp"

#include <climits>
#include <cstddef>
#include <functional>
#include <queue>
#include <utility>
#include <vector>

const int muhina_m_dijkstra_seq::TestTaskSequential::kEndOfVertexList = -1;

bool muhina_m_dijkstra_seq::TestTaskSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  graph_data_.assign(in_ptr, in_ptr + input_size);

  num_vertices_ = task_data->outputs_count[0];
  distances_.resize(num_vertices_);
  for (size_t i = 0; i < num_vertices_; ++i) {
    distances_[i] = INT_MAX;
  }
  if (task_data->inputs.size() > 1 && task_data->inputs[1] != nullptr) {
    start_vertex_ = *reinterpret_cast<int *>(task_data->inputs[1]);
  } else {
    start_vertex_ = 0;
  }
  distances_[start_vertex_] = 0;

  return true;
}

bool muhina_m_dijkstra_seq::TestTaskSequential::ValidationImpl() {
  return !task_data->inputs_count.empty() && task_data->inputs_count[0] > 0;
  return !task_data->outputs_count.empty() && task_data->outputs_count[0] > 0;
}

bool muhina_m_dijkstra_seq::TestTaskSequential::RunImpl() {
  std::vector<std::vector<std::pair<size_t, int>>> adj_list(num_vertices_);
  size_t current_vertex = 0;
  size_t i = 0;

  while (i < graph_data_.size() && current_vertex < num_vertices_) {
    if (graph_data_[i] == kEndOfVertexList) {
      current_vertex++;
      i++;
      continue;
    }

    if (i + 1 >= graph_data_.size()) {
      break;
    }

    size_t dest = graph_data_[i];
    int weight = graph_data_[i + 1];
    if (weight < 0) {
      return false;
    }

    if (dest < num_vertices_) {
      adj_list[current_vertex].emplace_back(dest, weight);
    }

    i += 2;
  }

  std::priority_queue<std::pair<int, size_t>, std::vector<std::pair<int, size_t>>, std::greater<>> pq;

  pq.emplace(0, start_vertex_);

  while (!pq.empty()) {
    size_t u = pq.top().second;
    int dist_u = pq.top().first;
    pq.pop();

    if (dist_u > distances_[u]) {
      continue;
    }

    for (const auto &edge : adj_list[u]) {
      size_t v = edge.first;
      int weight = edge.second;

      if (distances_[u] != INT_MAX && distances_[u] + weight < distances_[v]) {
        distances_[v] = distances_[u] + weight;
        pq.emplace(distances_[v], v);
      }
    }
  }

  return true;
}

bool muhina_m_dijkstra_seq::TestTaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < distances_.size(); ++i) {
    reinterpret_cast<int *>(task_data->outputs[0])[i] = distances_[i];
  }
  return true;
}
