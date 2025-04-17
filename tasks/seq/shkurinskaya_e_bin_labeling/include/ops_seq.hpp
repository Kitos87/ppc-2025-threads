#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shkurinskaya_e_bin_labeling {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int width_, height_;
  std::vector<int> input_;
  std::vector<int> res_;
  void Dfs(int x, int y, int comp);
};

}  // namespace shkurinskaya_e_bin_labeling
