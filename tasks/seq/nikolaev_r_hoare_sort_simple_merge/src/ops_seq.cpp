#include "../include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

bool nikolaev_r_hoare_sort_simple_merge_seq::HoareSortSimpleMergeSequential::PreProcessingImpl() {
  vect_size_ = task_data->inputs_count[0];
  auto *vect_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  vect_ = std::vector<double>(vect_ptr, vect_ptr + vect_size_);

  return true;
}

bool nikolaev_r_hoare_sort_simple_merge_seq::HoareSortSimpleMergeSequential::ValidationImpl() {
  return task_data->inputs_count[0] != 0 && task_data->outputs_count[0] != 0 && task_data->inputs[0] != nullptr &&
         task_data->outputs[0] != nullptr && task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool nikolaev_r_hoare_sort_simple_merge_seq::HoareSortSimpleMergeSequential::RunImpl() {
  QuickSort(0, vect_size_ - 1);
  return true;
}

bool nikolaev_r_hoare_sort_simple_merge_seq::HoareSortSimpleMergeSequential::PostProcessingImpl() {
  for (size_t i = 0; i < vect_size_; i++) {
    reinterpret_cast<double *>(task_data->outputs[0])[i] = vect_[i];
  }
  return true;
}

size_t nikolaev_r_hoare_sort_simple_merge_seq::HoareSortSimpleMergeSequential::Partition(size_t low, size_t high) {
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

void nikolaev_r_hoare_sort_simple_merge_seq::HoareSortSimpleMergeSequential::QuickSort(size_t low, size_t high) {
  if (low >= high) {
    return;
  }
  size_t pivot = Partition(low, high);
  if (pivot > low) {
    QuickSort(low, pivot - 1);
  }
  QuickSort(pivot + 1, high);
}