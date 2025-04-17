#include "seq/ermolaev_v_graham_scan/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

int ermolaev_v_graham_scan_seq::TestTaskSequential::CrossProduct(const Point &p1, const Point &p2, const Point &p3) {
  return ((p2.x - p1.x) * (p3.y - p1.y)) - ((p3.x - p1.x) * (p2.y - p1.y));
}

bool ermolaev_v_graham_scan_seq::TestTaskSequential::PreProcessingImpl() {
  auto *in_ptr = reinterpret_cast<Point *>(task_data->inputs[0]);
  input_ = std::vector<Point>(in_ptr, in_ptr + task_data->inputs_count[0]);
  output_ = std::vector<Point>();
  return true;
}

bool ermolaev_v_graham_scan_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] >= kMinInputPoints && task_data->inputs_count[0] <= task_data->outputs_count[0];
}

bool ermolaev_v_graham_scan_seq::TestTaskSequential::RunImpl() {
  {
    if (input_.size() < kMinInputPoints) {
      return false;
    }

    const Point &first = input_[0];
    bool all_same =
        std::ranges::all_of(input_.begin() + 1, input_.end(), [&first](const Point &p) { return p == first; });
    if (all_same) {
      return false;
    }

    bool found_non_collinear = false;
    for (size_t i = 0; i < input_.size() - 2 && !found_non_collinear; ++i) {
      for (size_t j = i + 1; j < input_.size() - 1 && !found_non_collinear; ++j) {
        for (size_t k = j + 1; k < input_.size() && !found_non_collinear; ++k) {
          if (CrossProduct(input_[i], input_[j], input_[k]) != 0) {
            found_non_collinear = true;
            break;
          }
        }
      }
    }

    if (!found_non_collinear) {
      return false;
    }
  }

  auto base_it = std::ranges::min_element(input_, [](const Point &a, const Point &b) { return a < b; });
  std::iter_swap(input_.begin(), base_it);

  std::sort(input_.begin() + 1, input_.end(), [&](const Point &a, const Point &b) {
    auto squared_dist = [](const Point &p1, const Point &p2) -> int {
      int dx = p1.x - p2.x;
      int dy = p1.y - p2.y;
      return ((dx * dx) + (dy * dy));
    };

    int cross = CrossProduct(input_[0], a, b);
    if (cross == 0) {
      return squared_dist(a, input_[0]) < squared_dist(b, input_[0]);
    }

    return cross > 0;
  });

  output_.clear();
  output_.emplace_back(input_[0]);
  output_.emplace_back(input_[1]);

  {
    Point p1;
    Point p2;
    Point p3;
    for (size_t i = kMinStackPoints; i < input_.size(); i++) {
      while (output_.size() >= kMinStackPoints) {
        p1 = output_[output_.size() - 2];
        p2 = output_[output_.size() - 1];
        p3 = input_[i];

        int cross = CrossProduct(p1, p2, p3);

        if (cross > 0) {
          break;
        }
        output_.pop_back();
      }
      output_.emplace_back(input_[i]);
    }
  }

  return true;
}

bool ermolaev_v_graham_scan_seq::TestTaskSequential::PostProcessingImpl() {
  task_data->outputs_count.clear();
  task_data->outputs_count.push_back(output_.size());
  std::ranges::copy(output_, reinterpret_cast<Point *>(task_data->outputs[0]));
  return true;
}
