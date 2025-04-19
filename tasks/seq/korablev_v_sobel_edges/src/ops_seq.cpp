#include "seq/korablev_v_sobel_edges/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

// clang-format off
constexpr int8_t kSobelKernelX[3][3] = {
  {-1, 0, 1},
  {-2, 0, 2},
  {-1, 0, 1}
};
constexpr int8_t kSobelKernelY[3][3] = {
  {-1, -2, -1},
  { 0,  0,  0},
  { 1,  2,  1}
};
// clang-format on

void korablev_v_sobel_edges_seq::Image::SetDimensions(std::size_t w,  // NOLINT(bugprone-easily-swappable-parameters)
                                                      std::size_t h) {
  width = w;
  height = h;
  data.resize(width * height * kPixelChannels);
}
void korablev_v_sobel_edges_seq::Image::CopyFrom(uint8_t* buf) {
  data.assign(buf, buf + (width * height * kPixelChannels));
}

bool korablev_v_sobel_edges_seq::TestTask::ValidationImpl() {
  const auto width = task_data->inputs_count[0];
  const auto height = task_data->inputs_count[1];
  return width > 0 && height > 0 && task_data->outputs_count[0] == (width * height * Image::kPixelChannels);
}

bool korablev_v_sobel_edges_seq::TestTask::PreProcessingImpl() {
  in_.SetDimensions(task_data->inputs_count[0], task_data->inputs_count[1]);
  in_.CopyFrom(task_data->inputs[0]);
  out_.SetDimensions(task_data->inputs_count[0], task_data->inputs_count[1]);
  return true;
}

bool korablev_v_sobel_edges_seq::TestTask::RunImpl() {
  const auto width = in_.width;
  const auto height = in_.height;

  auto& image = in_.data;

  for (std::size_t y = 1; y < height - 1; ++y) {
    for (std::size_t x = 1; x < width - 1; ++x) {
      std::array<int32_t, 3> sum_x{0};
      std::array<int32_t, 3> sum_y{0};

      for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
          int idx = ((y + ky) * width + (x + kx)) * 3;  // NOLINT(bugprone-narrowing-conversions)
          for (int j = 0; j < 3; j++) {
            sum_x[j] += kSobelKernelX[ky + 1][kx + 1] * image[idx + j];
            sum_y[j] += kSobelKernelY[ky + 1][kx + 1] * image[idx + j];
          }
        }
      }

      for (int i = 0; i < 3; ++i) {
        out_.data[((y * width + x) * 3) + i] = static_cast<uint8_t>(
            std::min(static_cast<int32_t>(std::sqrt((sum_x[i] * sum_x[i]) + (sum_y[i] * sum_y[i]))), 255));
      }
    }
  }

  return true;
}

bool korablev_v_sobel_edges_seq::TestTask::PostProcessingImpl() {
  std::ranges::copy(out_.data, task_data->outputs[0]);
  return true;
}
