#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace muhina_m_dijkstra_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> graph_data_;
  std::vector<int> distances_;
  size_t start_vertex_;
  size_t num_vertices_;
  static const int kEndOfVertexList;
};

}  // namespace muhina_m_dijkstra_seq