#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace tyshkevich_a_hoare_simple_merge_seq {

template <typename T, typename Comparator>
class HoareSortTask : public ppc::core::Task {
 public:
  explicit HoareSortTask(ppc::core::TaskDataPtr task_data, Comparator cmp) : Task(std::move(task_data)), cmp_(cmp) {}

  bool ValidationImpl() override { return task_data->inputs_count[0] == task_data->outputs_count[0]; }

  bool PreProcessingImpl() override {
    input_ = {reinterpret_cast<const T*>(task_data->inputs[0]), task_data->inputs_count[0]};
    output_ = {reinterpret_cast<T*>(task_data->outputs[0]), task_data->outputs_count[0]};
    return true;
  }

  bool RunImpl() override {
    std::copy(input_.begin(), input_.end(), output_.begin());
    HoareSort(output_, 0, output_.size() - 1);

    return true;
  }

  bool PostProcessingImpl() override {
    // output_ is being modified directly during Run
    return true;
  }

 private:
  void HoareSort(std::span<T> arr, int64_t low, int64_t high) {
    const auto partition = [&cmp = this->cmp_](std::span<T> region, int64_t plo, int64_t phi) -> int64_t {
      const auto& pivot = region[phi];
      int64_t e = plo - 1;
      for (int64_t j = plo; j <= phi - 1; j++) {
        if (cmp(region[j], pivot)) {
          std::swap(region[++e], region[j]);
        }
      }
      std::swap(region[e + 1], region[phi]);
      return e + 1;
    };

    if (low < high) {
      int64_t p = partition(arr, low, high);
      HoareSort(arr, low, p - 1);
      HoareSort(arr, p + 1, high);
    }
  };

  Comparator cmp_;

  std::span<const T> input_;
  std::span<T> output_;
};

template <typename T, typename Comparator>
HoareSortTask<T, Comparator> CreateHoareTestTask(ppc::core::TaskDataPtr task_data, Comparator cmp) {
  return HoareSortTask<T, Comparator>(std::move(task_data), cmp);
}

}  // namespace tyshkevich_a_hoare_simple_merge_seq