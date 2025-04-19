#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shuravina_o_hoare_simple_merger {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  std::vector<int> output_;

  void QuickSort(std::vector<int>& arr, int low, int high);
  static void Merge(std::vector<int>& arr, int low, int mid, int high);
};

}  // namespace shuravina_o_hoare_simple_merger