#include "seq/shkurinskaya_e_bin_labeling/include/ops_seq.hpp"

#include <algorithm>
#include <climits>
#include <vector>

bool shkurinskaya_e_bin_labeling::TestTaskSequential::PreProcessingImpl() {
  // Init value for input and output
  input_ = std::vector<int>(task_data->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  width_ = reinterpret_cast<int*>(task_data->inputs[1])[0];
  height_ = reinterpret_cast<int*>(task_data->inputs[2])[0];
  std::copy(tmp_ptr, tmp_ptr + task_data->inputs_count[0], input_.begin());
  // Init value for output
  res_.assign(task_data->inputs_count[0], 0);
  return true;
}

bool shkurinskaya_e_bin_labeling::TestTaskSequential::ValidationImpl() {
  // Check count elements of output
  return task_data->inputs_count[0] > 1 && task_data->outputs_count[0] == task_data->inputs_count[0] &&
         task_data->inputs_count[1] == 1 && task_data->inputs_count[2] == 1;
}

void shkurinskaya_e_bin_labeling::TestTaskSequential::Dfs(int x, int y, int comp) {
  if (x < 0 || x >= height_ || y < 0 || y >= width_) {
    return;
  }
  if (res_[(x * width_) + y] != 0 || input_[(x * width_) + y] != 1) {
    return;
  }
  res_[(x * width_) + y] = comp;
  for (int dx = -1; dx <= 1; dx++) {
    for (int dy = -1; dy <= 1; dy++) {
      if (dx == 0 && dy == 0) {
        continue;
      }
      Dfs(x + dx, y + dy, comp);
    }
  }
}

bool shkurinskaya_e_bin_labeling::TestTaskSequential::RunImpl() {
  int comp = 1;
  for (int i = 0; i < height_; ++i) {
    for (int j = 0; j < width_; ++j) {
      if (input_[(i * width_) + j] == 1 && res_[(i * width_) + j] == 0) {
        Dfs(i, j, comp++);
      }
    }
  }
  return true;
}

bool shkurinskaya_e_bin_labeling::TestTaskSequential::PostProcessingImpl() {
  std::ranges::copy(res_, reinterpret_cast<int*>(task_data->outputs[0]));
  return true;
}
