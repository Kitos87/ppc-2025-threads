#include "seq/chernykh_a_multidimensional_integral_rectangle/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

namespace chernykh_a_multidimensional_integral_rectangle_seq {

double Dimension::GetLowerBound() const { return lower_bound_; }

double Dimension::GetUpperBound() const { return upper_bound_; }

int Dimension::GetStepsCount() const { return steps_count_; }

double Dimension::GetStepSize() const { return (upper_bound_ - lower_bound_) / steps_count_; }

bool Dimension::IsValid() const { return lower_bound_ < upper_bound_ && steps_count_ > 0; }

bool SequentialTask::ValidationImpl() {
  auto *dims_ptr = reinterpret_cast<Dimension *>(task_data->inputs[0]);
  uint32_t dims_size = task_data->inputs_count[0];
  return dims_size > 0 &&
         std::all_of(dims_ptr, dims_ptr + dims_size, [](const Dimension &dim) -> bool { return dim.IsValid(); });
}

bool SequentialTask::PreProcessingImpl() {
  auto *dims_ptr = reinterpret_cast<Dimension *>(task_data->inputs[0]);
  uint32_t dims_size = task_data->inputs_count[0];
  dims_.assign(dims_ptr, dims_ptr + dims_size);
  return true;
}

bool SequentialTask::RunImpl() {
  int total_points = GetTotalPoints();
  auto point = Point(dims_.size());
  for (int i = 0; i < total_points; i++) {
    FillPoint(i, point);
    result_ += func_(point);
  }
  result_ *= GetScalingFactor();
  return true;
}

bool SequentialTask::PostProcessingImpl() {
  *reinterpret_cast<double *>(task_data->outputs[0]) = result_;
  return true;
}

void SequentialTask::FillPoint(int index, Point &point) const {
  for (size_t i = 0; i < dims_.size(); i++) {
    int coordinate_index = index % dims_[i].GetStepsCount();
    point[i] = dims_[i].GetLowerBound() + (coordinate_index + 1) * dims_[i].GetStepSize();
    index /= dims_[i].GetStepsCount();
  }
}

int SequentialTask::GetTotalPoints() const {
  return std::accumulate(dims_.begin(), dims_.end(), 1,
                         [](int accum, const Dimension &dim) -> int { return accum * dim.GetStepsCount(); });
}

double SequentialTask::GetScalingFactor() const {
  return std::accumulate(dims_.begin(), dims_.end(), 1.0,
                         [](double accum, const Dimension &dim) -> double { return accum * dim.GetStepSize(); });
}

}  // namespace chernykh_a_multidimensional_integral_rectangle_seq
