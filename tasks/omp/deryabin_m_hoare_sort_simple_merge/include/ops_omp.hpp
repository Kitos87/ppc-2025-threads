#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace deryabin_m_hoare_sort_simple_merge_omp {

// сортировка Хоара
void HoaraSort(std::vector<double>& a, size_t first, size_t last);
// слияние двух отсортированных частей
void MergeTwoParts(std::vector<double>& a, size_t left, size_t right, size_t dimension);

class HoareSortTaskSequential : public ppc::core::Task {
 public:
  explicit HoareSortTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_array_A_;  // входной массив
  size_t dimension_;                   // его размер
  size_t min_chunk_size_;  // размер частей на которые будет разбиваться исходный массив
  size_t chunk_count_;  // число таких частей
};
class HoareSortTaskOpenMP : public ppc::core::Task {
 public:
  explicit HoareSortTaskOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_array_A_;  // входной массив
  size_t dimension_;                   // его размер
  size_t min_chunk_size_;  // размер частей на которые будет разбиваться исходный массив
  size_t chunk_count_;  // число таких частей
};
}  // namespace deryabin_m_hoare_sort_simple_merge_omp
