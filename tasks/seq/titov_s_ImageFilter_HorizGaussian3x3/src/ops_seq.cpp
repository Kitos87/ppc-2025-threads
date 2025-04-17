#include "seq/titov_s_ImageFilter_HorizGaussian3x3/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

bool titov_s_image_filter_horiz_gaussian3x3_seq::ImageFilterSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  width_ = height_ = static_cast<int>(std::sqrt(input_size));
  input_.assign(in_ptr, in_ptr + input_size);

  auto *kernel_ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  kernel_ = std::vector<int>(kernel_ptr, kernel_ptr + 3);
  output_ = std::vector<double>(input_size, 0.0);

  return true;
}

bool titov_s_image_filter_horiz_gaussian3x3_seq::ImageFilterSequential::ValidationImpl() {
  auto *kernel_ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  kernel_ = std::vector<int>(kernel_ptr, kernel_ptr + 3);
  size_t size = input_.size();
  auto sqrt_size = static_cast<size_t>(std::sqrt(size));
  if (kernel_.size() != 3 || sqrt_size * sqrt_size != size) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool titov_s_image_filter_horiz_gaussian3x3_seq::ImageFilterSequential::RunImpl() {
  double sum = kernel_[0] + kernel_[1] + kernel_[2];
  for (int i = 0; i < height_; ++i) {
    for (int j = 0; j < width_; ++j) {
      double filtered_value = 0.0;
      for (int k = -1; k <= 1; ++k) {
        int col = j + k;
        if (col >= 0 && col < width_) {
          filtered_value += input_[(i * width_) + col] * kernel_[k + 1];
        }
      }

      output_[(i * width_) + j] = filtered_value / sum;
    }
  }

  return true;
}

bool titov_s_image_filter_horiz_gaussian3x3_seq::ImageFilterSequential::PostProcessingImpl() {
  auto *out_ptr = reinterpret_cast<double *>(task_data->outputs[0]);

  for (size_t i = 0; i < output_.size(); i++) {
    out_ptr[i] = output_[i];
  }

  return true;
}
