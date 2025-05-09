#include "all/korovin_n_qsort_batcher/include/ops_all.hpp"

#include <omp.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <random>
#include <ranges>
#include <span>
#include <vector>

#include "core/util/include/util.hpp"

namespace korovin_n_qsort_batcher_all {

int TestTaskALL::GetRandomIndex(int low, int high) {
  static thread_local std::mt19937 gen(std::random_device{}());
  std::uniform_int_distribution<int> dist(low, high);
  return dist(gen);
}

void TestTaskALL::QuickSort(std::vector<int>::iterator low, std::vector<int>::iterator high, int depth) {
  if (std::distance(low, high) <= 1) {
    return;
  }
  int n = static_cast<int>(std::distance(low, high));
  int random_index = GetRandomIndex(0, n - 1);
  auto pivot_iter = low + random_index;
  int pivot = *pivot_iter;
  auto partition_iter = std::partition(low, high, [pivot](int elem) { return elem <= pivot; });
  auto mid_iter = std::partition(low, partition_iter, [pivot](int elem) { return elem < pivot; });

  int max_depth = static_cast<int>(std::log2(ppc::util::GetPPCNumThreads())) + 1;

  if (depth < max_depth) {
#pragma omp parallel sections
    {
#pragma omp section
      { QuickSort(low, mid_iter, depth + 1); }
#pragma omp section
      { QuickSort(partition_iter, high, depth + 1); }
    }
  } else {
    QuickSort(low, mid_iter, depth + 1);
    QuickSort(partition_iter, high, depth + 1);
  }
}

bool TestTaskALL::InPlaceMerge(const BlockRange& a, const BlockRange& b, std::vector<int>& buffer) {
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

std::vector<BlockRange> TestTaskALL::PartitionBlocks(std::vector<int>& arr, int p) {
  std::vector<BlockRange> blocks;
  blocks.reserve(p);
  int n = static_cast<int>(arr.size());
  int chunk_size = n / p;
  int remainder = n % p;
  auto it = arr.begin();
  for (int i : std::views::iota(0, p)) {
    int size = chunk_size + (i < remainder ? 1 : 0);
    blocks.push_back({it, it + size});
    it += size;
  }
  return blocks;
}

void TestTaskALL::OddEvenMerge(std::vector<BlockRange>& blocks) {
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
  for (int iter : std::views::iota(0, max_iters)) {
    bool changed_global = false;
#pragma omp parallel for schedule(static) reduction(|| : changed_global)
    for (int b = iter % 2; b < p; b += 2) {
      static thread_local std::vector<int> buffer;
      if ((int)buffer.size() < buffer_size) {
        buffer.resize(buffer_size);
      }
      if (b + 1 < p) {
        bool changed_local = InPlaceMerge(blocks[b], blocks[b + 1], buffer);
        changed_global = changed_global || changed_local;
      }
    }
    if (!changed_global) {
      break;
    }
  }
}

bool TestTaskALL::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_.assign(in_ptr, in_ptr + input_size);
  return true;
}

bool TestTaskALL::ValidationImpl() {
  return (!task_data->inputs.empty()) && (!task_data->outputs.empty()) &&
         (task_data->inputs_count[0] == task_data->outputs_count[0]);
}

bool TestTaskALL::RunImpl() {
  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();
  if (size == 0) {
    return true;
  }
  int n = static_cast<int>(input_.size());
  std::vector<int> counts(size);
  std::vector<int> displs(size);
  int chunk = n / size;
  int rem = n % size;
  for (int i = 0; i < size; i++) {
    counts[i] = chunk + (i < rem ? 1 : 0);
    displs[i] = (i == 0 ? 0 : displs[i - 1] + counts[i - 1]);
  }
  std::vector<int> local(counts[rank]);
  int dummy = 0;
  bool have_local = counts[rank] != 0;
  int* local_ptr = have_local ? local.data() : &dummy;
  const int* root_ptr = input_.empty() ? &dummy : input_.data();
  boost::mpi::scatterv(world, root_ptr, counts, displs, local_ptr, counts[rank], 0);
  int p = std::max(ppc::util::GetPPCNumThreads(), 1);
  auto blocks = PartitionBlocks(local, p);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < p; i++) {
    QuickSort(blocks[i].low, blocks[i].high, 0);
  }
  if (rank == 0) {
    boost::mpi::gatherv(world, local_ptr, counts[rank], input_.data(), counts, displs, 0);
  } else {
    boost::mpi::gatherv(world, local_ptr, counts[rank], 0);
  }
  if (rank == 0) {
    int p2 = std::max(omp_get_max_threads(), 1);
    auto blocks2 = PartitionBlocks(input_, p2);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < p2; i++) {
      QuickSort(blocks2[i].low, blocks2[i].high, 0);
    }
    OddEvenMerge(blocks2);
  }
  return true;
}

bool TestTaskALL::PostProcessingImpl() {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    std::ranges::copy(input_, reinterpret_cast<int*>(task_data->outputs[0]));
  }
  return true;
}

}  // namespace korovin_n_qsort_batcher_all
