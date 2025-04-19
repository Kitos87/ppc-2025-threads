#include "omp/sotskov_a_shell_sorting_with_simple_merging/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <vector>

void sotskov_a_shell_sorting_with_simple_merging_omp::ShellSort(std::vector<int>& arr, int left, int right) {
  int array_size = right - left + 1;

  int gap = 1;
  while (gap < array_size / 3) {
    gap = gap * 3 + 1;
  }

  while (gap > 0) {
    for (int i = left + gap; i <= right; ++i) {
      int current_element = arr[i];
      int j = i;

      while (j >= left + gap && arr[j - gap] > current_element) {
        arr[j] = arr[j - gap];
        j -= gap;
      }
      arr[j] = current_element;
    }
    gap /= 3;
  }
}

void sotskov_a_shell_sorting_with_simple_merging_omp::ParallelMerge(std::vector<int>& arr, int left, int mid, int right,
                                                                    std::vector<int>& temp_buffer) {
  int i = left;
  int j = mid + 1;
  int k = 0;
  const int merge_size = right - left + 1;

  if (temp_buffer.size() < static_cast<std::size_t>(merge_size)) {
    temp_buffer.resize(static_cast<std::size_t>(merge_size));
  }

  while (i <= mid && j <= right) {
    temp_buffer[k++] = (arr[i] < arr[j]) ? arr[i++] : arr[j++];
  }

  while (i <= mid) {
    temp_buffer[k++] = arr[i++];
  }

  while (j <= right) {
    temp_buffer[k++] = arr[j++];
  }

  std::ranges::copy(temp_buffer.begin(), temp_buffer.begin() + k, arr.begin() + left);
}

void sotskov_a_shell_sorting_with_simple_merging_omp::ShellSortWithSimpleMerging(std::vector<int>& arr) {
  int array_size = static_cast<int>(arr.size());
  int num_threads = omp_get_max_threads();
  int chunk_size = (array_size + num_threads - 1) / num_threads;

#pragma omp parallel
  {
    std::vector<int> temp_buffer(chunk_size * 2);

#pragma omp for schedule(dynamic)
    for (int i = 0; i < num_threads; ++i) {
      int left = i * chunk_size;
      int right = std::min(left + chunk_size - 1, array_size - 1);

      if (left < right) {
        ShellSort(arr, left, right);
      }
    }

    for (int size = chunk_size; size < array_size; size *= 2) {
#pragma omp for schedule(dynamic)
      for (int left = 0; left < array_size; left += 2 * size) {
        int mid = std::min(left + size - 1, array_size - 1);
        int right = std::min(left + (2 * size) - 1, array_size - 1);

        if (mid < right) {
          ParallelMerge(arr, left, mid, right, temp_buffer);
        }
      }
    }
  }
}

bool sotskov_a_shell_sorting_with_simple_merging_omp::TestTaskOpenMP::PreProcessingImpl() {
  input_ = std::vector<int>(reinterpret_cast<int*>(task_data->inputs[0]),
                            reinterpret_cast<int*>(task_data->inputs[0]) + task_data->inputs_count[0]);

  return true;
}

bool sotskov_a_shell_sorting_with_simple_merging_omp::TestTaskOpenMP::ValidationImpl() {
  std::size_t input_size = task_data->inputs_count[0];
  std::size_t output_size = task_data->outputs_count[0];

  return (input_size == output_size);
}

bool sotskov_a_shell_sorting_with_simple_merging_omp::TestTaskOpenMP::RunImpl() {
  ShellSortWithSimpleMerging(input_);
  return true;
}

bool sotskov_a_shell_sorting_with_simple_merging_omp::TestTaskOpenMP::PostProcessingImpl() {
  int* output = reinterpret_cast<int*>(task_data->outputs[0]);
  std::ranges::copy(input_.begin(), input_.end(), output);
  return true;
}
