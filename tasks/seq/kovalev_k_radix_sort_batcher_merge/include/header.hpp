#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kovalev_k_radix_sort_batcher_merge_seq {

class RadixSortBatcherMerge : public ppc::core::Task {
 private:
  std::vector<long long int> mas_, tmp_;
  unsigned int n_;

 public:
  explicit RadixSortBatcherMerge(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool RadixUnsigned(unsigned long long*, unsigned long long*) const;
  bool Countbyte(unsigned long long*, int*, unsigned int) const;
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};
}  // namespace kovalev_k_radix_sort_batcher_merge_seq