#include "../include/ops_seq.hpp"

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace {
double AbsBound(const polikanov_v_rectangles::IntegrationBound& bound) { return bound.second - bound.first; }

std::size_t SizePow(std::size_t a, std::size_t n) {
  std::size_t r = 1;
  for (std::size_t i = 0; i < n; i++) {
    r *= a;
  }
  return r;
}
}  // namespace

class PointsIterator {
 public:
  PointsIterator(std::size_t discretization, std::vector<polikanov_v_rectangles::IntegrationBound> bounds)
      : discretization_(discretization),
        bounds_(std::move(bounds)),
        points_(SizePow(discretization, bounds_.size())),
        point(bounds_.size()) {
    Update();
  }

  bool Next() {
    ++cur_;
    Update();
    return cur_ < points_;
  }

 private:
  std::size_t discretization_;
  std::vector<std::pair<double, double>> bounds_;
  std::size_t points_;
  std::size_t cur_ = 0;

  void Update() {
    std::size_t c = cur_;
    auto it = point.begin();
    for (const auto& bound : bounds_) {
      *it = bound.first +
            (static_cast<double>(c % discretization_) * AbsBound(bound) / static_cast<double>(discretization_));
      ++it;
      c /= discretization_;
    }
  }

 public:
  std::vector<double> point;
};

namespace polikanov_v_rectangles {

bool polikanov_v_rectangles::TaskSEQ::ValidationImpl() {
  return task_data->inputs.size() == 2 && task_data->inputs_count[0] > 0 && task_data->outputs.size() == 1;
}

bool polikanov_v_rectangles::TaskSEQ::PreProcessingImpl() {
  const auto* bounds_ptr = reinterpret_cast<polikanov_v_rectangles::IntegrationBound*>(task_data->inputs[0]);
  const auto bounds_size = task_data->inputs_count[0];
  bounds_.assign(bounds_ptr, bounds_ptr + bounds_size);

  discretization_ = *reinterpret_cast<std::size_t*>(task_data->inputs[1]);

  return true;
}

bool polikanov_v_rectangles::TaskSEQ::RunImpl() {
  PointsIterator iter(discretization_, bounds_);

  result_ = 0.;
  do {
    result_ += function_(iter.point);
  } while (iter.Next());
  for (const auto& bound : bounds_) {
    result_ *= AbsBound(bound) / static_cast<double>(discretization_);
  }

  return true;
}

bool polikanov_v_rectangles::TaskSEQ::PostProcessingImpl() {
  *reinterpret_cast<double*>(task_data->outputs[0]) = result_;
  return true;
}
}  // namespace polikanov_v_rectangles
