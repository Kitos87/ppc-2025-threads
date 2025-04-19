#include "../include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

bool nikolaev_r_hoare_sort_simple_merge_omp::HoareSortSimpleMergeOpenMP::PreProcessingImpl() {
  vect_size_ = task_data->inputs_count[0];
  auto *vect_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  vect_ = std::vector<double>(vect_ptr, vect_ptr + vect_size_);

  return true;
}

bool nikolaev_r_hoare_sort_simple_merge_omp::HoareSortSimpleMergeOpenMP::ValidationImpl() {
  return task_data->inputs_count[0] != 0 && task_data->outputs_count[0] != 0 && task_data->inputs[0] != nullptr &&
         task_data->outputs[0] != nullptr && task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool nikolaev_r_hoare_sort_simple_merge_omp::HoareSortSimpleMergeOpenMP::RunImpl() {
  int num_threads = omp_get_max_threads();
  if (vect_size_ < static_cast<size_t>(num_threads)) {
    num_threads = static_cast<int>(vect_size_);
  }

  std::vector<std::pair<size_t, size_t>> segments;
  segments.reserve(num_threads);

  size_t seg_size = vect_size_ / num_threads;
  size_t remainder = vect_size_ % num_threads;
  size_t start = 0;
  for (int i = 0; i < num_threads; ++i) {
    size_t extra = (i < static_cast<int>(remainder) ? 1 : 0);
    size_t end = start + seg_size + extra - 1;
    segments.emplace_back(start, end);
    start = end + 1;
  }

#pragma omp parallel for schedule(static)
  for (int i = 0; i < static_cast<int>(segments.size()); i++) {
    QuickSort(segments[i].first, segments[i].second);
  }

  while (segments.size() > 1) {
    size_t num_pairs = segments.size() / 2;
    std::vector<std::pair<size_t, size_t>> new_segments;
    new_segments.resize(num_pairs);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(num_pairs); i++) {
      size_t left_start = segments[2 * i].first;
      size_t left_end = segments[2 * i].second;

      size_t right_start = segments[(2 * i) + 1].first;
      size_t right_end = segments[(2 * i) + 1].second;

      size_t merged_size = right_end - left_start + 1;
      std::vector<double> merged(merged_size);

      size_t i1 = left_start;
      size_t i2 = right_start;
      size_t k = 0;
      while (i1 <= left_end && i2 <= right_end) {
        if (vect_[i1] < vect_[i2]) {
          merged[k++] = vect_[i1++];
        } else {
          merged[k++] = vect_[i2++];
        }
      }
      while (i1 <= left_end) {
        merged[k++] = vect_[i1++];
      }
      while (i2 <= right_end) {
        merged[k++] = vect_[i2++];
      }

      std::ranges::copy(merged, vect_.begin() + static_cast<std::ptrdiff_t>(left_start));
      new_segments[i] = {left_start, right_end};
    }
    if (segments.size() % 2 == 1) {
      new_segments.push_back(segments.back());
    }
    segments.swap(new_segments);
  }
  return true;
}

bool nikolaev_r_hoare_sort_simple_merge_omp::HoareSortSimpleMergeOpenMP::PostProcessingImpl() {
  for (size_t i = 0; i < vect_size_; i++) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = vect_[i];
  }
  return true;
}

size_t nikolaev_r_hoare_sort_simple_merge_omp::HoareSortSimpleMergeOpenMP::Partition(size_t low, size_t high) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(static_cast<int>(low), static_cast<int>(high));

  size_t random_pivot_index = dist(gen);
  double pivot = vect_[random_pivot_index];

  std::swap(vect_[random_pivot_index], vect_[low]);
  size_t i = low + 1;

  for (size_t j = low + 1; j <= high; ++j) {
    if (vect_[j] < pivot) {
      std::swap(vect_[i], vect_[j]);
      i++;
    }
  }

  std::swap(vect_[low], vect_[i - 1]);
  return i - 1;
}

void nikolaev_r_hoare_sort_simple_merge_omp::HoareSortSimpleMergeOpenMP::QuickSort(size_t low, size_t high) {
  if (low >= high) {
    return;
  }
  size_t pivot = Partition(low, high);
  if (pivot > low) {
    QuickSort(low, pivot - 1);
  }
  QuickSort(pivot + 1, high);
}