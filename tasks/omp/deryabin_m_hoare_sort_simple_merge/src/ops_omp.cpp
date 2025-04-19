#include "omp/deryabin_m_hoare_sort_simple_merge/include/ops_omp.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

void deryabin_m_hoare_sort_simple_merge_omp::HoaraSort(std::vector<double>& a, size_t first, size_t last) {
  size_t i = first;
  size_t j = last;
  double tmp = 0;
  double x =
      std::max(std::min(a[first], a[(first + last) / 2]),
               std::min(std::max(a[first], a[(first + last) / 2]),
                        a[last]));  // выбор опорного элемента как медианы первого, среднего и последнего элементов
  do {
    while (a[i] < x) {
      i++;
    }
    while (a[j] > x) {
      j--;
    }
    if (i < j && a[i] > a[j]) {
      tmp = a[i];
      a[i] = a[j];
      a[j] = tmp;
    }
  } while (i < j);
  if (i < last) {
    HoaraSort(a, i + 1, last);
  }
  if (first < j) {
    HoaraSort(a, first, j);
  }
}

void deryabin_m_hoare_sort_simple_merge_omp::MergeTwoParts(std::vector<double>& a, size_t left, size_t right,
                                                           size_t dimension) {
  size_t middle = (right - left) / 2;
  size_t l_cur = 0;
  size_t r_cur = 0;
  std::vector<double> l_buff(middle + 1);
  std::vector<double> r_buff(middle + 1);
  std::copy(a.begin() + (long)left, a.begin() + (long)left + (long)middle + 1, l_buff.begin());
  std::copy(a.begin() + (long)left + (long)middle + 1, a.begin() + (long)right + 1, r_buff.begin());
  for (size_t i = left; i <= right; i++) {
    if (l_cur <= middle && r_cur <= middle) {
      if (l_buff[l_cur] < r_buff[r_cur]) {
        a[i] = l_buff[l_cur];
        l_cur++;
      } else {
        a[i] = r_buff[r_cur];
        r_cur++;
      }
    } else if (l_cur <= middle) {
      a[i] = l_buff[l_cur];
      l_cur++;
    } else {
      a[i] = r_buff[r_cur];
      r_cur++;
    }
  }
}

bool deryabin_m_hoare_sort_simple_merge_omp::HoareSortTaskSequential::PreProcessingImpl() {
  input_array_A_ = reinterpret_cast<std::vector<double>*>(task_data->inputs[0])[0];
  dimension_ = task_data->inputs_count[0];
  chunk_count_ = task_data->inputs_count[1];
  min_chunk_size_ = dimension_ / chunk_count_;
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_omp::HoareSortTaskSequential::ValidationImpl() {
  return static_cast<unsigned short>(task_data->inputs_count[0]) > 2 &&
         static_cast<unsigned short>(task_data->inputs_count[1]) >= 2 &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool deryabin_m_hoare_sort_simple_merge_omp::HoareSortTaskSequential::RunImpl() {
  size_t count = 0;
  size_t chunk_count = chunk_count_;
  while (count != chunk_count_) {
    HoaraSort(input_array_A_, count * min_chunk_size_, ((count + 1) * min_chunk_size_) - 1);
    count++;
  }
  for (size_t i = 0; i < (size_t)(log((double)chunk_count_) / std::numbers::ln2); i++) {
    for (size_t j = 0; j < chunk_count; j++) {
      MergeTwoParts(input_array_A_, j * min_chunk_size_ << (i + 1), ((j + 1) * min_chunk_size_ << (i + 1)) - 1,
                    dimension_);
      chunk_count--;
    }
  }
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_omp::HoareSortTaskSequential::PostProcessingImpl() {
  reinterpret_cast<std::vector<double>*>(task_data->outputs[0])[0] = input_array_A_;
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_omp::HoareSortTaskOpenMP::PreProcessingImpl() {
  input_array_A_ = reinterpret_cast<std::vector<double>*>(task_data->inputs[0])[0];
  dimension_ = task_data->inputs_count[0];
  chunk_count_ = task_data->inputs_count[1];
  min_chunk_size_ = dimension_ / chunk_count_;
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_omp::HoareSortTaskOpenMP::ValidationImpl() {
  return static_cast<unsigned short>(task_data->inputs_count[0]) > 2 &&
         static_cast<unsigned short>(task_data->inputs_count[1]) >= 2 &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool deryabin_m_hoare_sort_simple_merge_omp::HoareSortTaskOpenMP::RunImpl() {
  auto chunk_count = (short)chunk_count_;
#pragma omp parallel for
  for (short count = 0; count < chunk_count; count++) {
    HoaraSort(input_array_A_, count * (short)min_chunk_size_, ((count + 1) * (short)min_chunk_size_) - 1);
  }
#pragma omp barrier
#pragma omp parallel for ordered
  for (short i = 0; i < (short)(log((double)chunk_count_) / std::numbers::ln2); i++) {
#pragma omp ordered
    {
      for (short j = 0; j < chunk_count; j++) {
        MergeTwoParts(input_array_A_, j * (short)min_chunk_size_ << (i + 1),
                      ((j + 1) * (short)min_chunk_size_ << (i + 1)) - 1, dimension_);
        chunk_count--;
      }
    }
  }
#pragma omp barrier
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_omp::HoareSortTaskOpenMP::PostProcessingImpl() {
  reinterpret_cast<std::vector<double>*>(task_data->outputs[0])[0] = input_array_A_;
  return true;
}
