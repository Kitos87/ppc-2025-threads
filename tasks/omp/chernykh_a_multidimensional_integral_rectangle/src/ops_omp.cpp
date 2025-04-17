#include "omp/chernykh_a_multidimensional_integral_rectangle/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

namespace chernykh_a_multidimensional_integral_rectangle_omp {

double Dimension::GetLowerBound() const { return lower_bound_; }

double Dimension::GetUpperBound() const { return upper_bound_; }

int Dimension::GetStepsCount() const { return steps_count_; }

double Dimension::GetStepSize() const { return (upper_bound_ - lower_bound_) / steps_count_; }

bool Dimension::IsValid() const { return lower_bound_ < upper_bound_ && steps_count_ > 0; }

bool OMPTask::ValidationImpl() {
  auto *dims_ptr = reinterpret_cast<Dimension *>(task_data->inputs[0]);
  uint32_t dims_size = task_data->inputs_count[0];
  return dims_size > 0 &&
         std::all_of(dims_ptr, dims_ptr + dims_size, [](const Dimension &dim) -> bool { return dim.IsValid(); });
}

bool OMPTask::PreProcessingImpl() {
  auto *dims_ptr = reinterpret_cast<Dimension *>(task_data->inputs[0]);
  uint32_t dims_size = task_data->inputs_count[0];
  dims_.assign(dims_ptr, dims_ptr + dims_size);
  return true;
}

bool OMPTask::RunImpl() {
  double sum = 0.0;
  int total_points = GetTotalPoints();
#pragma omp parallel
  {
    auto thread_point = Point(dims_.size());
#pragma omp for reduction(+ : sum)
    for (int i = 0; i < total_points; i++) {
      FillPoint(i, thread_point);
      sum += func_(thread_point);
    }
  }
  result_ = sum * GetScalingFactor();
  return true;
}

bool OMPTask::PostProcessingImpl() {
  *reinterpret_cast<double *>(task_data->outputs[0]) = result_;
  return true;
}

void OMPTask::FillPoint(int index, Point &point) const {
  for (size_t i = 0; i < dims_.size(); i++) {
    int coordinate_index = index % dims_[i].GetStepsCount();
    point[i] = dims_[i].GetLowerBound() + (coordinate_index + 1) * dims_[i].GetStepSize();
    index /= dims_[i].GetStepsCount();
  }
}

int OMPTask::GetTotalPoints() const {
  return std::accumulate(dims_.begin(), dims_.end(), 1,
                         [](int accum, const Dimension &dim) -> int { return accum * dim.GetStepsCount(); });
}

double OMPTask::GetScalingFactor() const {
  return std::accumulate(dims_.begin(), dims_.end(), 1.0,
                         [](double accum, const Dimension &dim) -> double { return accum * dim.GetStepSize(); });
}

}  // namespace chernykh_a_multidimensional_integral_rectangle_omp
