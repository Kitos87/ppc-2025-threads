#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace tsatsyn_a_radix_sort_simple_merge_seq {
std::vector<double> GetRandomVector(int sz, int a, int b);
std::pair<std::vector<uint64_t>, std::vector<uint64_t>> ParseOrigin(std::vector<double> &input_data);
int CalculateBits(const std::vector<uint64_t> &data, bool is_pozitive);
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_data_;
  std::vector<double> output_;
};

}  // namespace tsatsyn_a_radix_sort_simple_merge_seq