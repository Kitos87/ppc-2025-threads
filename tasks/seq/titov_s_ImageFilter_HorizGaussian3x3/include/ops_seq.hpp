#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace titov_s_image_filter_horiz_gaussian3x3_seq {

class ImageFilterSequential : public ppc::core::Task {
 public:
  explicit ImageFilterSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_;
  std::vector<double> output_;
  int width_;
  int height_;
  int kernel_size_ = 3;
  std::vector<int> kernel_;

  void ApplyGaussianFilter();
};

}  // namespace titov_s_image_filter_horiz_gaussian3x3_seq
