#pragma once

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <compare>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace ermolaev_v_graham_scan_omp {

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

class TestTaskOMP : public ppc::core::Task {
 public:
  explicit TestTaskOMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<Point> input_, output_;

  static int CrossProduct(const Point& p1, const Point& p2, const Point& p3);
  size_t IndexOfMinElement();
  bool CheckGrahamNecessaryConditions();
  void GrahamScan();
  bool IsAllCollinear();
  bool IsAllSame();

  template <typename Iterator, typename Comparator>
  void ParallelSort(Iterator begin, Iterator end, Comparator comp);
};

}  // namespace ermolaev_v_graham_scan_omp

template <typename Iterator, typename Comparator>
void ermolaev_v_graham_scan_omp::TestTaskOMP::ParallelSort(Iterator begin, Iterator end, Comparator comp) {
  const size_t n = std::distance(begin, end);

  if (n <= 1) {
    return;
  }

  const int num_threads = omp_get_max_threads();
  std::vector<std::vector<typename Iterator::value_type>> sorted_chunks(num_threads);

#pragma omp parallel
  {
    const int thread_id = omp_get_thread_num();
    const size_t chunk_size = n / num_threads;
    const size_t start_idx = thread_id * chunk_size;
    const size_t end_idx = (thread_id == num_threads - 1) ? n : start_idx + chunk_size;

    if (start_idx < n) {
      std::vector<typename Iterator::value_type> chunk(begin + start_idx, begin + end_idx);
      std::sort(chunk.begin(), chunk.end(), comp);
      sorted_chunks[thread_id] = std::move(chunk);
    }
  }

  std::vector<size_t> sizes(num_threads, 0);
  std::transform(sorted_chunks.begin(), sorted_chunks.end(), sizes.begin(),
                 [](const auto& chunk) { return chunk.size(); });

  const int levels = static_cast<int>(std::ceil(std::log2(num_threads)));
  std::vector<typename Iterator::value_type> temp;
  int source = 0;

  for (int level = 0; level < levels; level++) {
    int step = 1 << level;
    int h = step << 1;

#pragma omp parallel for schedule(dynamic) private(temp, source)
    for (int i = 0; i < num_threads; i += h) {
      source = i + step;
      if (source < num_threads && sizes[i] > 0 && sizes[source] > 0) {
        temp.clear();
        temp.reserve(sizes[i] + sizes[source]);

        std::merge(sorted_chunks[i].begin(), sorted_chunks[i].end(), sorted_chunks[source].begin(),
                   sorted_chunks[source].end(), std::back_inserter(temp), comp);

        sorted_chunks[i].swap(temp);

        sizes[i] += sizes[source];
        sizes[source] = 0;
      }
    }
  }

  for (int i = 0; i < num_threads; i++) {
    if (sizes[i] > 0) {
      std::copy(sorted_chunks[i].begin(), sorted_chunks[i].end(), begin);
      return;
    }
  }
}