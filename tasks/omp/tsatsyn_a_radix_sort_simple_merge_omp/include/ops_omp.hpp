#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace tsatsyn_a_radix_sort_simple_merge_omp {
inline int CalculateBits(const std::vector<uint64_t> &data, bool is_pozitive);
inline std::vector<uint64_t> MainSort(std::vector<uint64_t> &data, int bit);

class TestTaskOpenMP : public ppc::core::Task {
 public:
  explicit TestTaskOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_data_;
  std::vector<double> output_;
};

}  // namespace tsatsyn_a_radix_sort_simple_merge_omp