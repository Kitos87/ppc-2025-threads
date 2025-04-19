#include <cstdlib>
#include <cstring>
#include <vector>

#include "seq/kovalev_k_radix_sort_batcher_merge/include/header.hpp"

bool kovalev_k_radix_sort_batcher_merge_seq::RadixSortBatcherMerge::RadixUnsigned(unsigned long long* inp_arr,
                                                                                  unsigned long long* mas_tmp) const {
  auto* masc = reinterpret_cast<unsigned char*>(inp_arr);
  int count[256];
  unsigned int sizetype = sizeof(unsigned long long);
  for (unsigned int i = 0; i < sizetype; i++) {
    Countbyte(inp_arr, count, i);
    for (unsigned int j = 0; j < n_; j++) {
      mas_tmp[count[masc[(j * sizetype) + i]]++] = inp_arr[j];
    }
    memcpy(inp_arr, mas_tmp, sizeof(unsigned long long) * n_);
  }
  return true;
}

bool kovalev_k_radix_sort_batcher_merge_seq::RadixSortBatcherMerge::Countbyte(unsigned long long* inp_arr, int* count,
                                                                              unsigned int byte) const {
  auto* masc = reinterpret_cast<unsigned char*>(inp_arr);
  unsigned int bias = sizeof(unsigned long long);
  for (unsigned int i = 0; i < 256; i++) {
    count[i] = 0;
  }
  for (unsigned int i = 0; i < n_; i++) {
    count[masc[(i * bias) + byte]]++;
  }
  int tmp1 = count[0];
  count[0] = 0;
  for (unsigned int i = 1; i < 256; i++) {
    int tmp2 = count[i];
    count[i] = count[i - 1] + tmp1;
    tmp1 = tmp2;
  }
  return true;
}

bool kovalev_k_radix_sort_batcher_merge_seq::RadixSortBatcherMerge::ValidationImpl() {
  return (task_data->inputs_count[0] > 0 && task_data->outputs_count[0] == task_data->inputs_count[0]);
}

bool kovalev_k_radix_sort_batcher_merge_seq::RadixSortBatcherMerge::PreProcessingImpl() {
  n_ = task_data->inputs_count[0];
  mas_ = std::vector<long long int>(n_);
  tmp_ = std::vector<long long int>(n_);
  void* ptr_input = task_data->inputs[0];
  void* ptr_vec = mas_.data();
  memcpy(ptr_vec, ptr_input, sizeof(long long int) * n_);
  return true;
}

bool kovalev_k_radix_sort_batcher_merge_seq::RadixSortBatcherMerge::RunImpl() {
  unsigned int count = 0;
  bool ret = RadixUnsigned(reinterpret_cast<unsigned long long*>(mas_.data()),
                           reinterpret_cast<unsigned long long*>(tmp_.data()));
  while (count < n_ && mas_[count] >= 0) {
    count++;
  }
  if (count == n_) {
    return ret;
  }
  memcpy(tmp_.data(), mas_.data() + count, sizeof(long long int) * (n_ - count));
  memcpy(tmp_.data() + (n_ - count), mas_.data(), sizeof(long long int) * (count));
  memcpy(mas_.data(), tmp_.data(), sizeof(long long int) * n_);
  return ret;
}

bool kovalev_k_radix_sort_batcher_merge_seq::RadixSortBatcherMerge::PostProcessingImpl() {
  memcpy(reinterpret_cast<long long int*>(task_data->outputs[0]), mas_.data(), sizeof(long long int) * n_);
  return true;
}