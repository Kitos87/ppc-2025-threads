#include "stl/korovin_n_qsort_batcher/include/ops_stl.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <random>
#include <span>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace korovin_n_qsort_batcher_stl {

int TestTaskSTL::GetRandomIndex(int low, int high) {
  static thread_local std::mt19937 gen(std::random_device{}());
  std::uniform_int_distribution<int> dist(low, high);
  return dist(gen);
}

void TestTaskSTL::QuickSort(std::vector<int>::iterator low, std::vector<int>::iterator high, int depth) {
  if (std::distance(low, high) <= 1) {
    return;
  }
  int n = static_cast<int>(std::distance(low, high));
  int pivot_index = GetRandomIndex(0, n - 1);
  auto pivot_iter = low + pivot_index;
  int pivot = *pivot_iter;
  auto mid = std::partition(low, high, [pivot](int elem) { return elem <= pivot; });
  auto mid_iter = std::partition(low, mid, [pivot](int elem) { return elem < pivot; });
  QuickSort(low, mid_iter, depth + 1);
  QuickSort(mid, high, depth + 1);
}

bool TestTaskSTL::InPlaceMerge(const BlockRange& a, const BlockRange& b, std::vector<int>& buffer) {
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

std::vector<BlockRange> TestTaskSTL::PartitionBlocks(std::vector<int>& arr, int p) {
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

void TestTaskSTL::OddEvenMerge(std::vector<BlockRange>& blocks) {
  if (blocks.size() <= 1) {
    return;
  }
  int p = static_cast<int>(blocks.size());
  int max_iters = p * 2;
  int max_block_len = 0;
  for (auto& b : blocks) {
    max_block_len = std::max(max_block_len, static_cast<int>(std::distance(b.low, b.high)));
  }
  int buffer_size = max_block_len * 2;
  for (int iter = 0; iter < max_iters; iter++) {
    std::atomic<bool> changed_global(false);
    std::vector<std::thread> threads;
    for (int b = iter % 2; b < p; b += 2) {
      threads.emplace_back([&blocks, &changed_global, b, buffer_size]() {
        static thread_local std::vector<int> buffer;
        if (static_cast<int>(buffer.size()) < buffer_size) {
          buffer.resize(buffer_size);
        }
        if (b + 1 < static_cast<int>(blocks.size())) {
          bool changed_local = InPlaceMerge(blocks[b], blocks[b + 1], buffer);
          if (changed_local) {
            changed_global.store(true, std::memory_order_relaxed);
          }
        }
      });
    }
    for (auto& t : threads) {
      t.join();
    }
    if (!changed_global.load(std::memory_order_relaxed)) {
      break;
    }
  }
}

bool TestTaskSTL::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_.assign(in_ptr, in_ptr + input_size);
  return true;
}

bool TestTaskSTL::ValidationImpl() {
  return (!task_data->inputs.empty()) && (!task_data->outputs.empty()) &&
         (task_data->inputs_count[0] == task_data->outputs_count[0]);
}

bool TestTaskSTL::RunImpl() {
  int n = static_cast<int>(input_.size());
  if (n <= 1) {
    return true;
  }
  int num_threads = ppc::util::GetPPCNumThreads();
  int tasks = std::min(num_threads, std::max(1, n / 256));
  auto blocks = PartitionBlocks(input_, tasks);
  std::vector<std::thread> threads;
  threads.reserve(tasks);
  for (int i = 0; i < tasks; i++) {
    threads.emplace_back([i, &blocks]() { QuickSort(blocks[i].low, blocks[i].high, 0); });
  }
  for (auto& t : threads) {
    t.join();
  }
  OddEvenMerge(blocks);
  return true;
}

bool TestTaskSTL::PostProcessingImpl() {
  std::ranges::copy(input_, reinterpret_cast<int*>(task_data->outputs[0]));
  return true;
}

}  // namespace korovin_n_qsort_batcher_stl
