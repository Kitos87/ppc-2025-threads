#include "seq/voroshilov_v_convex_hull_components/include/chc_seq.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include "seq/voroshilov_v_convex_hull_components/include/chc.hpp"

using namespace voroshilov_v_convex_hull_components_seq;

bool voroshilov_v_convex_hull_components_seq::ChcTaskSequential::ValidationImpl() {
  int *ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  int height = *ptr;
  ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  int width = *ptr;
  return height > 0 && width > 0;
}

bool voroshilov_v_convex_hull_components_seq::ChcTaskSequential::PreProcessingImpl() {
  int *ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  int height = *ptr;

  ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  int width = *ptr;

  std::vector<int> pixels(task_data->inputs_count[0]);
  ptr = reinterpret_cast<int *>(task_data->inputs[2]);
  std::ranges::copy(ptr, ptr + task_data->inputs_count[0], pixels.begin());

  Image image(height, width, pixels);
  imageIn_ = image;

  return true;
}

bool voroshilov_v_convex_hull_components_seq::ChcTaskSequential::RunImpl() {
  std::vector<Component> components = FindComponents(imageIn_);

  hullsOut_ = QuickHullAll(components);

  return true;
}

bool voroshilov_v_convex_hull_components_seq::ChcTaskSequential::PostProcessingImpl() {
  std::pair<std::vector<int>, std::vector<int>> packed_out = PackHulls(hullsOut_, imageIn_);
  std::vector<int> hulls_indexes = packed_out.first;
  std::vector<int> pixels_indexes = packed_out.second;

  std::ranges::copy(hulls_indexes, reinterpret_cast<int *>(task_data->outputs[0]));
  std::ranges::copy(pixels_indexes, reinterpret_cast<int *>(task_data->outputs[1]));
  task_data->outputs_count[0] = hullsOut_.size();

  return true;
}
