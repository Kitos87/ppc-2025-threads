#include "seq/kozlova_e_contrast_enhancement/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

bool kozlova_e_contrast_enhancement_seq::TestTaskSequential::PreProcessingImpl() {
  auto* input_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  size_t size = task_data->inputs_count[0];
  width_ = task_data->inputs_count[1];
  height_ = task_data->inputs_count[2];
  output_.resize(size, 0);
  input_.resize(size);
  std::copy(input_ptr, input_ptr + size, input_.begin());

  return true;
}

bool kozlova_e_contrast_enhancement_seq::TestTaskSequential::ValidationImpl() {
  size_t size = task_data->inputs_count[0];
  size_t check_width = task_data->inputs_count[1];
  size_t check_height = task_data->inputs_count[2];
  return size == task_data->outputs_count[0] && size > 0 && (size % 2 == 0) && check_width >= 1 && check_height >= 1 &&
         (size == check_height * check_width);
}

bool kozlova_e_contrast_enhancement_seq::TestTaskSequential::RunImpl() {
  int min_value = *std::ranges::min_element(input_);
  if (min_value < 0) {
    throw "incorrect value";
  }
  int max_value = *std::ranges::max_element(input_);

  if (min_value == max_value) {
    std::ranges::copy(input_, output_.begin());
    return true;
  }

  for (size_t i = 0; i < input_.size(); ++i) {
    output_[i] = static_cast<int>(((input_[i] - min_value) / (double)(max_value - min_value)) * 255);
    output_[i] = std::clamp(output_[i], 0, 255);
  }

  return true;
}

bool kozlova_e_contrast_enhancement_seq::TestTaskSequential::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); ++i) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
