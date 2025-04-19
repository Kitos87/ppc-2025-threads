#pragma once

#include <cmath>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kudryashova_i_radix_batcher_seq {
std::vector<double> GetRandomDoubleVector(int size);
void RadixDoubleSort(std::vector<double> &data, int first, int last);
class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_data_;
};

}  // namespace kudryashova_i_radix_batcher_seq