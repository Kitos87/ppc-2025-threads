#include "omp/varfolomeev_g_histogram_linear_stretching/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

bool varfolomeev_g_histogram_linear_stretching_omp::TestTaskSequential::PreProcessingImpl() {
  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<uint8_t *>(task_data->inputs[0]);
  img_ = std::vector<uint8_t>(in_ptr, in_ptr + input_size);
  res_.resize(img_.size());
  return true;
}

bool varfolomeev_g_histogram_linear_stretching_omp::TestTaskSequential::ValidationImpl() {
  return task_data->outputs_count[0] > 0 && task_data->inputs_count[0] > 0 &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool varfolomeev_g_histogram_linear_stretching_omp::TestTaskSequential::RunImpl() {
  uint8_t min = *std::ranges::min_element(img_);
  uint8_t max = *std::ranges::max_element(img_);

  if (max != min) {
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(img_.size()); i++) {
      res_[i] = static_cast<uint8_t>(((img_[i] - min) * 255 + (max - min) / 2) / (max - min));
    }
  } else {
    std::ranges::fill(res_.begin(), res_.end(), min);
  }
  return true;
}

bool varfolomeev_g_histogram_linear_stretching_omp::TestTaskSequential::PostProcessingImpl() {
  std::memcpy(task_data->outputs[0], res_.data(), res_.size() * sizeof(uint8_t));

  return true;
}