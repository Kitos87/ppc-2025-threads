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
  thread_local static std::mt19937 gen(std::random_device{}());
  std::uniform_int_distribution<int> dist(low, high);
  return dist(gen);
}

void TestTaskSTL::QuickSort(std::vector<int>::iterator low, std::vector<int>::iterator high, int depth) {
  int n = static_cast<int>(std::distance(low, high));
  if (n <= 1) {
    return;
  }

  int random_index = GetRandomIndex(0, n - 1);
  int pivot = *(low + random_index);

  auto partition_iter = std::partition(low, high, [pivot](int elem) { return elem <= pivot; });
  auto mid_iter = std::partition(low, partition_iter, [pivot](int elem) { return elem < pivot; });

  int max_threads = ppc::util::GetPPCNumThreads();

  int current_threads = thread_count_.load(std::memory_order_relaxed);
  bool spawn_thread = false;
  if (current_threads < max_threads &&
      thread_count_.compare_exchange_weak(current_threads, current_threads + 1, std::memory_order_relaxed)) {
    spawn_thread = true;
  }

  if (spawn_thread) {
    std::thread th([low, mid_iter]() {
      QuickSort(low, mid_iter, 0);
      thread_count_.fetch_sub(1, std::memory_order_relaxed);
    });
    QuickSort(partition_iter, high, 0);
    th.join();
  } else {
    QuickSort(low, mid_iter, 0);
    QuickSort(partition_iter, high, 0);
  }
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
  for (const auto& b : blocks) {
    int len = static_cast<int>(std::distance(b.low, b.high));
    max_block_len = std::max(max_block_len, len);
  }
  int buffer_size = max_block_len * 2;
  std::vector<std::vector<int>> buffers(p / 2, std::vector<int>(buffer_size));
  std::vector<std::thread> threads;
  threads.reserve(p / 2);
  std::vector<bool> changed_local_vec((p + 1) / 2);

  for (int iter = 0; iter < max_iters; iter++) {
    bool changed_global = false;
    std::fill(changed_local_vec.begin(), changed_local_vec.end(), false);
    threads.clear();
    for (int i = iter % 2; i + 1 < p; i += 2) {
      int pair_idx = i / 2;
      threads.emplace_back([&, i, pair_idx]() {
        bool changed_local = InPlaceMerge(blocks[i], blocks[i + 1], buffers[pair_idx]);
        changed_local_vec[pair_idx] = changed_local;
      });
    }
    for (auto& thread : threads) {
      thread.join();
    }
    for (bool c : changed_local_vec) {
      if (c) {
        changed_global = true;
        break;
      }
    }
    if (!changed_global) {
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
  int p = std::max(num_threads / 2, 1);
  auto blocks = PartitionBlocks(input_, p);

  std::vector<std::thread> threads;
  threads.reserve(p);
  for (int i = 0; i < p; i++) {
    threads.emplace_back([&, i] { QuickSort(blocks[i].low, blocks[i].high, 0); });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  OddEvenMerge(blocks);
  return true;
}

bool TestTaskSTL::PostProcessingImpl() {
  std::ranges::copy(input_, reinterpret_cast<int*>(task_data->outputs[0]));
  return true;
}

}  // namespace korovin_n_qsort_batcher_stl
