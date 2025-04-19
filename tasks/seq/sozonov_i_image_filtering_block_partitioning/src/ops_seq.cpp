#include "seq/sozonov_i_image_filtering_block_partitioning/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

bool sozonov_i_image_filtering_block_partitioning_seq::TestTaskSequential::PreProcessingImpl() {
  // Init image
  image_ = std::vector<double>(task_data->inputs_count[0]);
  auto *image_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  std::ranges::copy(image_ptr, image_ptr + task_data->inputs_count[0], image_.begin());

  width_ = static_cast<int>(task_data->inputs_count[1]);
  height_ = static_cast<int>(task_data->inputs_count[2]);

  // Init filtered image
  filtered_image_ = std::vector<double>(width_ * height_, 0);
  return true;
}

bool sozonov_i_image_filtering_block_partitioning_seq::TestTaskSequential::ValidationImpl() {
  // Init image
  image_ = std::vector<double>(task_data->inputs_count[0]);
  auto *image_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  std::ranges::copy(image_ptr, image_ptr + task_data->inputs_count[0], image_.begin());

  size_t img_size = task_data->inputs_count[1] * task_data->inputs_count[2];

  // Check pixels range from 0 to 255
  for (size_t i = 0; i < img_size; ++i) {
    if (image_[i] < 0 || image_[i] > 255) {
      return false;
    }
  }

  // Check size of image
  return task_data->inputs_count[0] > 0 && task_data->inputs_count[0] == img_size &&
         task_data->outputs_count[0] == img_size && task_data->inputs_count[1] >= 3 && task_data->inputs_count[2] >= 3;
}

bool sozonov_i_image_filtering_block_partitioning_seq::TestTaskSequential::RunImpl() {
  // Linear image filtering
  std::vector<double> kernel = {0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};
  for (int i = 1; i < height_ - 1; ++i) {
    for (int j = 1; j < width_ - 1; ++j) {
      double sum = 0;
      for (int l = -1; l <= 1; ++l) {
        for (int k = -1; k <= 1; ++k) {
          sum += image_[((i - l) * width_) + j - k] * kernel[((l + 1) * 3) + k + 1];
        }
      }
      filtered_image_[(i * width_) + j] = sum;
    }
  }
  return true;
}

bool sozonov_i_image_filtering_block_partitioning_seq::TestTaskSequential::PostProcessingImpl() {
  auto *out = reinterpret_cast<double *>(task_data->outputs[0]);
  std::ranges::copy(filtered_image_.begin(), filtered_image_.end(), out);
  return true;
}
