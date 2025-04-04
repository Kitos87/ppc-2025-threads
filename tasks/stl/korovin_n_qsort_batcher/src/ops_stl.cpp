#include "stl/korovin_n_qsort_batcher/include/ops_stl.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iterator>
#include <random>
#include <span>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace korovin_n_qsort_batcher_stl {

int TestTaskSTL::GetRandomIndex(int low, int high) {
  thread_local static std::mt19937 gen(123456789u);
  std::uniform_int_distribution<int> dist(low, high);
  return dist(gen);
}

void TestTaskSTL::QuickSort(std::vector<int>::iterator low, std::vector<int>::iterator high, int depth) {
  if (std::distance(low, high) <= 1) {
    return;
  }

  int random_index = GetRandomIndex(0, std::distance(low, high) - 1);
  int pivot = *(low + random_index);

  auto partition_iter = std::partition(low, high, [pivot](int elem) { return elem <= pivot; });
  auto mid_iter = std::partition(low, partition_iter, [pivot](int elem) { return elem < pivot; });

  int max_depth = static_cast<int>(std::log2(ppc::util::GetPPCNumThreads())) + 1;

  if (depth < max_depth) {
    std::thread left(QuickSort, low, mid_iter, depth + 1);
    std::thread right(QuickSort, partition_iter, high, depth + 1);
    left.join();
    right.join();
  } else {
    QuickSort(low, mid_iter, depth + 1);
    QuickSort(partition_iter, high, depth + 1);
  }
}

bool TestTaskSTL::InPlaceMerge(const BlockRange& a, const BlockRange& b, std::vector<int>& buffer) {
  bool changed = false;

  std::span<int> span_a(a.low, a.high);
  std::span<int> span_b(b.low, b.high);

  size_t i = 0, j = 0, k = 0;

  while (i < span_a.size() && j < span_b.size()) {
    if (span_a[i] <= span_b[j]) {
      buffer[k++] = span_a[i++];
    } else {
      changed = true;
      buffer[k++] = span_b[j++];
    }
  }
  while (i < span_a.size()) buffer[k++] = span_a[i++];
  while (j < span_b.size()) {
    changed = true;
    buffer[k++] = span_b[j++];
  }

  std::copy(buffer.begin(), buffer.begin() + span_a.size(), a.low);
  std::copy(buffer.begin() + span_a.size(), buffer.begin() + span_a.size() + span_b.size(), b.low);

  return changed;
}

std::vector<BlockRange> TestTaskSTL::PartitionBlocks(std::vector<int>& arr, int p) {
  std::vector<BlockRange> blocks;
  blocks.reserve(p);
  int chunk_size = arr.size() / p;
  int remainder = arr.size() % p;

  auto it = arr.begin();
  for (int i = 0; i < p; i++) {
    int size = chunk_size + (i < remainder ? 1 : 0);
    blocks.push_back({it, it + size});
    it += size;
  }
  return blocks;
}

void TestTaskSTL::OddEvenMerge(std::vector<BlockRange>& blocks) {
  if (blocks.size() <= 1) return;

  int p = blocks.size();
  int max_iters = p * 2;

  int max_block_len = 0;
  for (const auto& b : blocks) {
    int len = std::distance(b.low, b.high);
    max_block_len = std::max(max_block_len, len);
  }
  int buffer_size = max_block_len * 2;

  std::vector<std::vector<int>> buffers(p / 2, std::vector<int>(buffer_size));

  for (int iter = 0; iter < max_iters; iter++) {
    std::atomic<bool> changed_global(false);
    std::vector<std::thread> threads;

    for (int i = iter % 2; i + 1 < p; i += 2) {
      threads.emplace_back([&, i]() {
        bool changed_local = InPlaceMerge(blocks[i], blocks[i + 1], buffers[i / 2]);
        if (changed_local) changed_global.store(true, std::memory_order_relaxed);
      });
    }
    for (auto& thread : threads) {
      thread.join();
    }
    if (!changed_global.load()) {
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
  int n = (int)input_.size();
  if (n <= 1) return true;

  const int GRAIN_SIZE = 2000;
  int p_auto = (int)std::ceil((double)n / GRAIN_SIZE);

  int num_threads = ppc::util::GetPPCNumThreads();
  int p = std::min(p_auto, num_threads);

  if (p < 1) p = 1;

  auto blocks = PartitionBlocks(input_, p);

  std::vector<std::thread> threads;
  threads.reserve(p);
  for (int i = 0; i < p; i++) {
    threads.emplace_back([&, i]() { QuickSort(blocks[i].low, blocks[i].high, 0); });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  OddEvenMerge(blocks);
  return true;
}

bool TestTaskSTL::PostProcessingImpl() {
  std::copy(input_.begin(), input_.end(), reinterpret_cast<int*>(task_data->outputs[0]));
  return true;
}

}  // namespace korovin_n_qsort_batcher_stl
