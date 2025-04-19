#include "seq/milovankin_m_histogram_stretching/include/ops_seq.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace milovankin_m_histogram_stretching_seq {

bool TestTaskSequential::ValidationImpl() {
  return !task_data->inputs.empty() && !task_data->inputs_count.empty() && task_data->inputs_count[0] != 0 &&
         !task_data->outputs.empty() && !task_data->outputs_count.empty() &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool TestTaskSequential::PreProcessingImpl() {
  const uint8_t* input_data = task_data->inputs.front();
  const uint32_t input_size = task_data->inputs_count.front();

  img_.assign(input_data, input_data + input_size);
  return true;
}

bool TestTaskSequential::RunImpl() {
  auto minmax = std::ranges::minmax(img_);
  uint8_t min_val = minmax.min;
  uint8_t max_val = minmax.max;

  if (min_val != max_val) {
    const int delta = max_val - min_val;
    std::ranges::for_each(
        img_, [&min_val, &delta](uint8_t& pixel) { pixel = ((pixel - min_val) * 255 + delta / 2) / delta; });
  }

  return true;
}

bool TestTaskSequential::PostProcessingImpl() {
  uint8_t* output_data = task_data->outputs[0];
  const uint32_t output_size = task_data->outputs_count[0];
  const uint32_t copy_size = std::min(output_size, static_cast<uint32_t>(img_.size()));

  std::copy_n(img_.cbegin(), copy_size, output_data);
  return true;
}

}  // namespace milovankin_m_histogram_stretching_seq
