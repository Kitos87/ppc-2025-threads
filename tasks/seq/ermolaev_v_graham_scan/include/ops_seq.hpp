#pragma once

#include <compare>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace ermolaev_v_graham_scan_seq {

constexpr size_t kMinInputPoints = 3;
constexpr size_t kMinStackPoints = 2;

class Point {
 public:
  int x;
  int y;

  Point(int x_value, int y_value) : x(x_value), y(y_value) {}
  Point() : x(0), y(0) {}
  bool operator==(const Point& rhs) const { return y == rhs.y && x == rhs.x; }
  bool operator!=(const Point& rhs) const { return !(*this == rhs); }
  auto operator<=>(const Point& rhs) const {
    if (auto cmp = y <=> rhs.y; cmp != 0) {
      return cmp;
    }
    return x <=> rhs.x;
  }
};

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<Point> input_, output_;

  static int CrossProduct(const Point& p1, const Point& p2, const Point& p3);
};

}  // namespace ermolaev_v_graham_scan_seq