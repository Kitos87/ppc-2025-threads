#include "seq/malyshev_a_increase_contrast_by_histogram/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

bool malyshev_a_increase_contrast_by_histogram_seq::TestTaskSequential::PreProcessingImpl() {
  data_.assign(task_data->inputs[0], task_data->inputs[0] + task_data->inputs_count[0]);

  return !data_.empty();
}

bool malyshev_a_increase_contrast_by_histogram_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs[0] != nullptr && task_data->outputs[0] != nullptr && task_data->inputs_count.size() == 1 &&
         task_data->outputs_count.size() == 1 && task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool malyshev_a_increase_contrast_by_histogram_seq::TestTaskSequential::RunImpl() {
  auto [temp_min, temp_max] = std::ranges::minmax_element(data_);
  uint8_t min = *temp_min;
  uint8_t max = *temp_max;

  if (min == max) {
    return true;
  }

  std::ranges::for_each(data_, [&min, &max](uint8_t& pixel) { pixel = (pixel - min) * 255 / (max - min); });
  return true;
}

bool malyshev_a_increase_contrast_by_histogram_seq::TestTaskSequential::PostProcessingImpl() {
  std::ranges::copy(data_, task_data->outputs[0]);

  return true;
}
