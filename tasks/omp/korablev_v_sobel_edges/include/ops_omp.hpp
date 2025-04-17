#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace korablev_v_sobel_edges_omp {

struct Pixel {
  uint8_t r;
  uint8_t g;
  uint8_t b;
};

struct Image {
  std::size_t width;
  std::size_t height;
  std::vector<uint8_t> data;

  static constexpr auto kPixelChannels = 3;

  void SetDimensions(std::size_t w, std::size_t h);
  void CopyFrom(uint8_t* buf);
};

class TestTask : public ppc::core::Task {
 public:
  explicit TestTask(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  Image in_;
  Image out_;
};

}  // namespace korablev_v_sobel_edges_omp