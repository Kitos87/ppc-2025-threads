#include "seq/koshkin_n_shell_sort_batchers_even_odd_merge/include/ops_seq.hpp"

#include <algorithm>
#include <random>
#include <vector>

std::vector<int> koshkin_n_shell_sort_batchers_even_odd_merge_seq::GetRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(-100000, 100000);
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}

void koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential::Swap(std::vector<int> &a, int i, int j,
                                                                                bool order) {
  if ((order && a[i] > a[j]) || (!order && a[i] < a[j])) {
    std::swap(a[i], a[j]);
  }
}

void koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential::BatcherMerge(std::vector<int> &a, int start,
                                                                                        int end, bool order) {
  int distance = end - start;
  if (distance <= 1) {
    return;
  }

  int mid = start + (distance / 2);

  BatcherMerge(a, start, mid, order);
  BatcherMerge(a, mid, end, order);

  // проверка на отсортированность
  bool is_sorted = true;
  for (int i = start; i < end - 1; ++i) {
    if ((order && a[i] > a[i + 1]) || (!order && a[i] < a[i + 1])) {
      is_sorted = false;
      break;
    }
  }

  if (is_sorted) {
    return;
  }

  for (int i = start; i + 1 < end; i += 2) {
    Swap(a, i, i + 1, order);
  }
  for (int i = start + 1; i + 1 < end; i += 2) {
    Swap(a, i, i + 1, order);
  }
}

void koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential::ShellBatcherSort(std::vector<int> &a,
                                                                                            bool order) {
  int n = static_cast<int>(a.size());
  int gap = 1;

  // генерация шагов Кнута
  while (gap < n / 3) {
    gap = 3 * gap + 1;
  }

  // шаги Шелла
  while (gap > 0) {
    for (int i = gap; i < n; i++) {
      int temp = a[i];
      int j = 0;
      for (j = i; j >= gap && ((order && a[j - gap] > temp) || (!order && a[j - gap] < temp)); j -= gap) {
        a[j] = a[j - gap];
      }
      a[j] = temp;
    }
    gap /= 3;
  }

  // четно-нечетное слияние Бэтчера
  BatcherMerge(a, 0, n, order);
}

bool koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];

  input_ = std::vector<int>(input_size);

  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  std::ranges::copy(in_ptr, in_ptr + input_size, input_.begin());

  input_order_ = *reinterpret_cast<bool *>(task_data->inputs[1]);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);

  return true;
}

bool koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential::ValidationImpl() {
  return ((!task_data->inputs.empty() && !task_data->outputs.empty()) &&
          (!task_data->inputs_count.empty() && task_data->inputs_count[0] != 0) &&
          (!task_data->outputs_count.empty() && task_data->outputs_count[0] != 0));
}

bool koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential::RunImpl() {
  ShellBatcherSort(input_, input_order_);
  output_ = input_;
  return true;
}

bool koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential::PostProcessingImpl() {
  auto *output = reinterpret_cast<int *>(task_data->outputs[0]);

  std::ranges::copy(output_.begin(), output_.end(), output);
  return true;
}