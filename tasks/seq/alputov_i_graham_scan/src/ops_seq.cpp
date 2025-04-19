#include "seq/alputov_i_graham_scan/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

namespace alputov_i_graham_scan_seq {

Point::Point(double x, double y) : x(x), y(y) {}

bool Point::operator<(const Point& other) const { return std::tie(y, x) < std::tie(other.y, other.x); }

bool Point::operator==(const Point& other) const { return std::tie(x, y) == std::tie(other.x, other.y); }

bool TestTaskSequential::PreProcessingImpl() {
  auto* input_ptr = reinterpret_cast<Point*>(task_data->inputs[0]);
  input_points_ = std::vector<Point>(input_ptr, input_ptr + task_data->inputs_count[0]);
  return true;
}

bool TestTaskSequential::ValidationImpl() {
  return (task_data->inputs_count[0] <= task_data->outputs_count[0] && task_data->inputs_count[0] >= 3);
}

double TestTaskSequential::Cross(const Point& o, const Point& a, const Point& b) {
  return ((a.x - o.x) * (b.y - o.y)) - ((a.y - o.y) * (b.x - o.x));
}

Point TestTaskSequential::FindPivot() const {
  return *std::ranges::min_element(input_points_, [](const Point& a, const Point& b) { return a < b; });
}

std::vector<Point> TestTaskSequential::SortPoints(const Point& pivot) const {
  std::vector<Point> points = input_points_;
  auto [new_end, _] = std::ranges::remove(points, pivot);
  points.erase(new_end, points.end());

  std::ranges::sort(points, [](const Point& a, const Point& b) { return a < b; });
  auto [unique_end, discarded] = std::ranges::unique(points);
  points.erase(unique_end, points.end());

  std::ranges::sort(points, [&pivot](const Point& a, const Point& b) {
    const double angle_a = atan2(a.y - pivot.y, a.x - pivot.x);
    const double angle_b = atan2(b.y - pivot.y, b.x - pivot.x);
    return (angle_a < angle_b) || (angle_a == angle_b && a.x < b.x);
  });
  return points;
}
const std::vector<Point>& TestTaskSequential::GetConvexHull() const { return convex_hull_; }

std::vector<Point> TestTaskSequential::BuildHull(const std::vector<Point>& sorted_points) const {
  std::vector<Point> hull;
  hull.reserve(sorted_points.size());
  hull.push_back(FindPivot());
  hull.push_back(sorted_points[0]);
  hull.push_back(sorted_points[1]);

  for (size_t i = 2; i < sorted_points.size(); ++i) {
    while (hull.size() >= 2 && Cross(hull[hull.size() - 2], hull.back(), sorted_points[i]) <= 0) {
      hull.pop_back();
    }
    hull.push_back(sorted_points[i]);
  }
  return hull;
}

bool TestTaskSequential::RunImpl() {
  const Point pivot = FindPivot();
  const auto sorted_points = SortPoints(pivot);
  convex_hull_ = BuildHull(sorted_points);

  return true;
}

bool TestTaskSequential::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<Point*>(task_data->outputs[0]);
  std::ranges::copy(convex_hull_, output_ptr);
  return true;
}

}  // namespace alputov_i_graham_scan_seq