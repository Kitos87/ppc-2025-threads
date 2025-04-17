#include "seq/deryabin_m_hoare_sort_simple_merge/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

bool deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential::PreProcessingImpl() {
  input_array_A_ = reinterpret_cast<std::vector<double>*>(task_data->inputs[0])[0];
  dimension_ = task_data->inputs_count[0];
  chunk_count_ = task_data->inputs_count[1];
  min_chunk_size_ = dimension_ / chunk_count_;
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential::ValidationImpl() {
  return static_cast<unsigned short>(task_data->inputs_count[0]) > 2 &&
         static_cast<unsigned short>(task_data->inputs_count[1]) >= 2 &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

void deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential::HoaraSort(std::vector<double>& a, size_t first,
                                                                                size_t last) {
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

void deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential::MergeTwoParts(std::vector<double>& a, size_t left,
                                                                                    size_t right) const {
  size_t middle = left + ((right - left) / 2);
  size_t l_cur = left;
  size_t r_cur = middle + 1;
  std::vector<double> l_buff(dimension_);
  std::vector<double> r_buff(dimension_);
  std::copy(a.begin() + (long)l_cur, a.begin() + (long)r_cur, l_buff.begin() + (long)l_cur);
  std::copy(a.begin() + (long)r_cur, a.begin() + (long)right + 1, r_buff.begin() + (long)r_cur);
  for (size_t i = left; i <= right; i++) {
    if (l_cur <= middle && r_cur <= right) {
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

bool deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential::RunImpl() {
  size_t count = 0;
  size_t chunk_count = chunk_count_;
  while (count != chunk_count_) {
    HoaraSort(input_array_A_, count * min_chunk_size_, ((count + 1) * min_chunk_size_) - 1);
    count++;
  }
  for (size_t i = 0; i < (size_t)(log((double)chunk_count_) / std::numbers::ln2); i++) {
    for (size_t j = 0; j < chunk_count; j++) {
      MergeTwoParts(input_array_A_, j * min_chunk_size_ << (i + 1), ((j + 1) * min_chunk_size_ << (i + 1)) - 1);
      chunk_count--;
    }
  }
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential::PostProcessingImpl() {
  reinterpret_cast<std::vector<double>*>(task_data->outputs[0])[0] = input_array_A_;
  return true;
}
