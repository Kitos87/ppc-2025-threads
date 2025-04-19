#include "seq/korovin_n_qsort_batcher/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <random>
#include <vector>

int korovin_n_qsort_batcher_seq::TestTaskSequential::GetRandomIndex(int low, int high) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(low, high);
  return dist(gen);
}

void korovin_n_qsort_batcher_seq::TestTaskSequential::QuickSort(std::vector<int>& arr, int low, int high) {
  if (low >= high) {
    return;
  }

  int partition_index = GetRandomIndex(low, high);
  int partition_value = arr[partition_index];

  auto partition_iter = std::partition(arr.begin() + low, arr.begin() + high + 1,
                                       [partition_value](const int& elem) { return elem <= partition_value; });

  auto mid_iter = std::partition(arr.begin() + low, partition_iter,
                                 [partition_value](const int& elem) { return elem < partition_value; });

  int i = static_cast<int>(std::distance(arr.begin(), mid_iter));
  int j = static_cast<int>(std::distance(arr.begin(), partition_iter) - 1);

  if (low < i - 1) {
    QuickSort(arr, low, i - 1);
  }
  if (j + 1 < high) {
    QuickSort(arr, j + 1, high);
  }
}

bool korovin_n_qsort_batcher_seq::TestTaskSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_.assign(in_ptr, in_ptr + input_size);
  return true;
}

bool korovin_n_qsort_batcher_seq::TestTaskSequential::ValidationImpl() {
  return (!task_data->inputs.empty()) && (!task_data->outputs.empty()) &&
         (task_data->inputs_count[0] == task_data->outputs_count[0]);
}

bool korovin_n_qsort_batcher_seq::TestTaskSequential::RunImpl() {
  int n = static_cast<int>(input_.size());
  if (n <= 1) {
    return true;
  }
  QuickSort(input_, 0, n - 1);
  return true;
}

bool korovin_n_qsort_batcher_seq::TestTaskSequential::PostProcessingImpl() {
  std::ranges::copy(input_, reinterpret_cast<int*>(task_data->outputs[0]));
  return true;
}
