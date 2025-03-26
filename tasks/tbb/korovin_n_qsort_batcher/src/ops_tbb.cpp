#include "tbb/korovin_n_qsort_batcher/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <random>
#include <span>
#include <vector>

#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/enumerable_thread_specific.h"
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/task_group.h"

namespace korovin_n_qsort_batcher_tbb {

int TestTaskTBB::GetRandomIndex(int low, int high) {
  static thread_local std::mt19937 gen(std::random_device{}());
  std::uniform_int_distribution<int> dist(low, high);
  return dist(gen);
}

void TestTaskTBB::QuickSort(std::vector<int>::iterator low, std::vector<int>::iterator high, int depth) {
  if (std::distance(low, high) <= 1) {
    return;
  }

  int n = static_cast<int>(std::distance(low, high));
  int random_index = GetRandomIndex(0, n - 1);
  int pivot = *(low + random_index);

  auto partition_iter = std::partition(low, high, [pivot](int elem) { return elem <= pivot; });
  auto mid_iter = std::partition(low, partition_iter, [pivot](int elem) { return elem < pivot; });

  int max_depth = static_cast<int>(std::log2(tbb::this_task_arena::max_concurrency())) + 1;

  if (depth < max_depth) {
    tbb::task_group group;
    group.run([=] { QuickSort(low, mid_iter, depth + 1); });
    group.run([=] { QuickSort(partition_iter, high, depth + 1); });
    group.wait();
  } else {
    QuickSort(low, mid_iter, depth + 1);
    QuickSort(partition_iter, high, depth + 1);
  }
}

bool TestTaskTBB::InPlaceMerge(const BlockRange& a, const BlockRange& b, std::vector<int>& buffer) {
  bool changed = false;
  int len_a = static_cast<int>(std::distance(a.low, a.high));
  int len_b = static_cast<int>(std::distance(b.low, b.high));

  std::span<int> span_a{a.low, static_cast<size_t>(len_a)};
  std::span<int> span_b{b.low, static_cast<size_t>(len_b)};

  size_t i = 0;
  size_t j = 0;
  size_t k = 0;

  while (i < span_a.size() && j < span_b.size()) {
    if (span_a[i] <= span_b[j]) {
      buffer[k++] = span_a[i++];
    } else {
      changed = true;
      buffer[k++] = span_b[j++];
    }
  }
  while (i < span_a.size()) {
    buffer[k++] = span_a[i++];
  }
  while (j < span_b.size()) {
    changed = true;
    buffer[k++] = span_b[j++];
  }

  std::ranges::copy(buffer.begin(), buffer.begin() + len_a, a.low);
  std::ranges::copy(buffer.begin() + len_a, buffer.begin() + len_a + len_b, b.low);

  return changed;
}

std::vector<BlockRange> TestTaskTBB::PartitionBlocks(std::vector<int>& arr, int p) {
  std::vector<BlockRange> blocks;
  blocks.reserve(p);
  int n = static_cast<int>(arr.size());
  int chunk_size = n / p;
  int remainder = n % p;

  auto it = arr.begin();
  for (int i = 0; i < p; i++) {
    int size = chunk_size + (i < remainder ? 1 : 0);
    blocks.push_back({it, it + size});
    it += size;
  }
  return blocks;
}

void TestTaskTBB::OddEvenMerge(std::vector<BlockRange>& blocks) {
  if (blocks.size() <= 1) {
    return;
  }

  int p = static_cast<int>(blocks.size());
  int max_iters = p * 2;
  int max_block_len = 0;
  for (const auto& b : blocks) {
    int len = static_cast<int>(std::distance(b.low, b.high));
    max_block_len = std::max(max_block_len, len);
  }
  int buffer_size = max_block_len * 2;
  tbb::enumerable_thread_specific<std::vector<int>> thread_buffers;
  for (int iter = 0; iter < max_iters; iter++) {
    std::atomic<bool> changed_global = false;
    tbb::parallel_for(tbb::blocked_range<int>(iter % 2, p, 2), [&](const tbb::blocked_range<int>& range) {
      auto& buffer = thread_buffers.local();
      if (static_cast<int>(buffer.size()) < buffer_size) {
        buffer.resize(buffer_size);
      }
      for (int i = range.begin(); i < range.end(); i += 2) {
        if (i + 1 < p) {
          bool changed_local = InPlaceMerge(blocks[i], blocks[i + 1], buffer);
          if (changed_local) {
            changed_global.store(true, std::memory_order_relaxed);
          }
        }
      }
    });
    if (!changed_global.load()) {
      break;
    }
  }
}

bool TestTaskTBB::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_.assign(in_ptr, in_ptr + input_size);
  return true;
}

bool TestTaskTBB::ValidationImpl() {
  return (!task_data->inputs.empty()) && (!task_data->outputs.empty()) &&
         (task_data->inputs_count[0] == task_data->outputs_count[0]);
}

bool TestTaskTBB::RunImpl() {
  int n = static_cast<int>(input_.size());
  if (n <= 1) {
    return true;
  }
  int num_threads = tbb::this_task_arena::max_concurrency();
  int p = std::max(num_threads / 2, 1);
  auto blocks = PartitionBlocks(input_, p);

  tbb::parallel_for(0, p, [&](int i) { QuickSort(blocks[i].low, blocks[i].high, 0); });

  OddEvenMerge(blocks);
  return true;
}

bool TestTaskTBB::PostProcessingImpl() {
  std::ranges::copy(input_, reinterpret_cast<int*>(task_data->outputs[0]));
  return true;
}

}  // namespace korovin_n_qsort_batcher_tbb
