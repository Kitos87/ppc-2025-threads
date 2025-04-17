#include "seq/shuravina_o_hoare_simple_merger/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace shuravina_o_hoare_simple_merger {

void TestTaskSequential::QuickSort(std::vector<int>& arr, int low, int high) {
  if (low < high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; ++j) {
      if (arr[j] <= pivot) {
        ++i;
        std::swap(arr[i], arr[j]);
      }
    }
    std::swap(arr[i + 1], arr[high]);

    int pi = i + 1;

    QuickSort(arr, low, pi - 1);
    QuickSort(arr, pi + 1, high);
  }
}

void TestTaskSequential::Merge(std::vector<int>& arr, int low, int mid, int high) {
  std::vector<int> temp(high - low + 1);
  int i = low;
  int j = mid + 1;
  int k = 0;

  while (i <= mid && j <= high) {
    if (arr[i] <= arr[j]) {
      temp[k++] = arr[i++];
    } else {
      temp[k++] = arr[j++];
    }
  }

  while (i <= mid) {
    temp[k++] = arr[i++];
  }

  while (j <= high) {
    temp[k++] = arr[j++];
  }

  for (i = low, k = 0; i <= high; ++i, ++k) {
    arr[i] = temp[k];
  }
}

bool TestTaskSequential::PreProcessingImpl() {
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  auto input_size = static_cast<size_t>(task_data->inputs_count[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  auto output_size = static_cast<size_t>(task_data->outputs_count[0]);
  output_ = std::vector<int>(output_size, 0);

  return true;
}

bool TestTaskSequential::ValidationImpl() { return task_data->inputs_count[0] == task_data->outputs_count[0]; }

bool TestTaskSequential::RunImpl() {
  auto size = input_.size();
  QuickSort(input_, 0, static_cast<int>(size) - 1);
  Merge(input_, 0, static_cast<int>(size / 2) - 1, static_cast<int>(size) - 1);
  output_ = input_;
  return true;
}

bool TestTaskSequential::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
  for (size_t i = 0; i < output_.size(); i++) {
    out_ptr[i] = output_[i];
  }
  return true;
}

}  // namespace shuravina_o_hoare_simple_merger