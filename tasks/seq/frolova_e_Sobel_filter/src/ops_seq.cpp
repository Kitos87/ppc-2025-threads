#include "seq/frolova_e_Sobel_filter/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

std::vector<int> frolova_e_sobel_filter_seq::ToGrayScaleImg(std::vector<RGB>& color_img, size_t width, size_t height) {
  std::vector<int> gray_scale_image(width * height);
  for (size_t i = 0; i < width * height; i++) {
    gray_scale_image[i] =
        static_cast<int>((0.299 * color_img[i].R) + (0.587 * color_img[i].G) + (0.114 * color_img[i].B));
  }

  return gray_scale_image;
}

int frolova_e_sobel_filter_seq::Clamp(int value, int min_val, int max_val) {
  if (value < min_val) {
    return min_val;
  }
  if (value > max_val) {
    return max_val;
  }
  return value;
}

bool frolova_e_sobel_filter_seq::SobelFilterSequential::PreProcessingImpl() {
  int* value_1 = reinterpret_cast<int*>(task_data->inputs[0]);
  width_ = static_cast<size_t>(value_1[0]);

  height_ = static_cast<size_t>(value_1[1]);

  int* value_2 = reinterpret_cast<int*>(task_data->inputs[1]);
  std::vector<int> picture_vector;
  picture_vector.assign(value_2, value_2 + task_data->inputs_count[1]);
  for (size_t i = 0; i < picture_vector.size(); i += 3) {
    RGB pixel;
    pixel.R = picture_vector[i];
    pixel.G = picture_vector[i + 1];
    pixel.B = picture_vector[i + 2];

    picture_.push_back(pixel);
  }

  grayscale_image_ = frolova_e_sobel_filter_seq::ToGrayScaleImg(picture_, width_, height_);
  res_image_.resize(width_ * height_);

  return true;
}

bool frolova_e_sobel_filter_seq::SobelFilterSequential::ValidationImpl() {
  int* value_1 = reinterpret_cast<int*>(task_data->inputs[0]);

  if (task_data->inputs_count[0] != 2) {
    return false;
  }

  if (value_1[0] <= 0 || value_1[1] <= 0) {
    return false;
  }

  auto width_1 = static_cast<size_t>(value_1[0]);
  auto height_1 = static_cast<size_t>(value_1[1]);

  int* value_2 = reinterpret_cast<int*>(task_data->inputs[1]);
  std::vector<int> picture_vector;
  picture_vector.assign(value_2, value_2 + task_data->inputs_count[1]);
  if (task_data->inputs_count[1] != width_1 * height_1 * 3) {
    return false;
  }

  for (size_t i = 0; i < picture_vector.size(); i++) {
    if (picture_vector[i] < 0 || picture_vector[i] > 255) {
      return false;
    }
  }

  return true;
}

bool frolova_e_sobel_filter_seq::SobelFilterSequential::RunImpl() {
  const std::vector<int> gx = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  const std::vector<int> gy = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
  for (size_t y = 0; y < height_; y++) {
    for (size_t x = 0; x < width_; x++) {
      int res_x = 0;
      int res_y = 0;

      for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
          int px = static_cast<int>(x) + kx;
          int py = static_cast<int>(y) + ky;

          int pixel_value = 0;

          if (px >= 0 && px < static_cast<int>(width_) && py >= 0 && py < static_cast<int>(height_)) {
            pixel_value = grayscale_image_[(py * width_) + px];
          }

          size_t kernel_ind = ((ky + 1) * 3) + (kx + 1);
          res_x += pixel_value * gx[kernel_ind];
          res_y += pixel_value * gy[kernel_ind];
        }
      }
      int gradient = static_cast<int>(sqrt((res_x * res_x) + (res_y * res_y)));
      res_image_[(y * width_) + x] = Clamp(gradient, 0, 255);
    }
  }
  return true;
}

bool frolova_e_sobel_filter_seq::SobelFilterSequential::PostProcessingImpl() {
  for (size_t i = 0; i < width_ * height_; i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = res_image_[i];
  }
  return true;
}