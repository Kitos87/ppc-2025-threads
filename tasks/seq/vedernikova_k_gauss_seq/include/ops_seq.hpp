#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

using Kernel = std::vector<double>;
using Image = std::vector<uint8_t>;

namespace vedernikova_k_gauss_seq {

class Gauss : public ppc::core::Task {
 public:
  explicit Gauss(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  Image input_;
  Image output_;

  Kernel kernel_;

  uint32_t width_;
  uint32_t height_;
  uint32_t channels_;
  uint32_t size_;

  void ComputeKernel(double sigma = 5.0 / 12);
  uint8_t GetPixel(uint32_t x, uint32_t y, uint32_t channel);
  void SetPixel(uint8_t value, uint32_t x, uint32_t y, uint32_t channel);
  double GetMultiplier(int i, int j);
  void ComputePixel(uint32_t x, uint32_t y);
};

}  // namespace vedernikova_k_gauss_seq