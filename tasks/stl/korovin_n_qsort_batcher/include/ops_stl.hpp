#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace korovin_n_qsort_batcher_stl {

struct BlockRange {
  std::vector<int>::iterator low;
  std::vector<int>::iterator high;
};

class TestTaskSTL : public ppc::core::Task {
 public:
  explicit TestTaskSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  static int GetRandomIndex(int low, int high);
  static void QuickSort(std::vector<int>::iterator low, std::vector<int>::iterator high, int depth = 0);
  static bool InPlaceMerge(const BlockRange& a, const BlockRange& b, std::vector<int>& buffer);
  static std::vector<BlockRange> PartitionBlocks(std::vector<int>& arr, int p);
  static void OddEvenMerge(std::vector<BlockRange>& blocks);
};

}  // namespace korovin_n_qsort_batcher_stl
