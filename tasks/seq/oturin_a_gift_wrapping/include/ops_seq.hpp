#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace oturin_a_gift_wrapping_seq {

struct Coord {
  int x, y;
  bool operator==(const Coord o) const { return (x == o.x && y == o.y); }
  bool operator!=(const Coord o) const { return x != o.x || y != o.y; }
};

double Distance(Coord a, Coord b);

// Angle Between Three Points
double ABTP(Coord a, Coord b, Coord c);

// Angle Between Three Points for leftmost point
double ABTP(Coord a, Coord c);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<Coord> input_, output_;
  int n_;

  int FindMostLeft();
  void PointSearch(double t, double &line_angle, int &search_index, int i);
};

}  // namespace oturin_a_gift_wrapping_seq