#include "seq/plekhanov_d_dijkstra/include/ops_seq.hpp"

#include <climits>
#include <cstddef>
#include <set>
#include <utility>
#include <vector>

const int plekhanov_d_dijkstra_seq::TestTaskSequential::kEndOfVertexList = -1;

bool plekhanov_d_dijkstra_seq::TestTaskSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  graph_data_.assign(in_ptr, in_ptr + input_size);
  num_vertices_ = task_data->outputs_count[0];
  distances_.resize(num_vertices_);
  distances_.assign(num_vertices_, INT_MAX);
  if (task_data->inputs.size() > 1 && task_data->inputs[1] != nullptr) {
    start_vertex_ = *reinterpret_cast<int *>(task_data->inputs[1]);
  } else {
    start_vertex_ = 0;
  }
  distances_[start_vertex_] = 0;
  return true;
}

bool plekhanov_d_dijkstra_seq::TestTaskSequential::ValidationImpl() {
  return !task_data->inputs_count.empty() && task_data->inputs_count[0] > 0 && !task_data->outputs_count.empty() &&
         task_data->outputs_count[0] > 0;
}

bool plekhanov_d_dijkstra_seq::TestTaskSequential::RunImpl() {
  std::vector<std::vector<std::pair<size_t, int>>> adj_list(num_vertices_);
  const size_t estimated_edges_per_vertex = 8;
  for (auto &edges : adj_list) {
    edges.reserve(estimated_edges_per_vertex);
  }
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
  distances_[start_vertex_] = 0;
  std::set<std::pair<int, size_t>> vertex_set;
  vertex_set.insert({0, start_vertex_});
  while (!vertex_set.empty()) {
    size_t u = vertex_set.begin()->second;
    vertex_set.erase(vertex_set.begin());
    for (const auto &[v, weight] : adj_list[u]) {
      if (distances_[u] != INT_MAX && distances_[u] + weight < distances_[v]) {
        if (distances_[v] != INT_MAX) {
          vertex_set.erase({distances_[v], v});
        }
        distances_[v] = distances_[u] + weight;
        vertex_set.insert({distances_[v], v});
      }
    }
  }
  return true;
}

bool plekhanov_d_dijkstra_seq::TestTaskSequential::PostProcessingImpl() {
  auto *output = reinterpret_cast<int *>(task_data->outputs[0]);
  for (size_t i = 0; i < distances_.size(); ++i) {
    output[i] = distances_[i];
  }
  return true;
}
