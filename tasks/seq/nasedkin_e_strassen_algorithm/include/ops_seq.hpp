#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace nasedkin_e_strassen_algorithm_seq {

std::vector<double> StandardMultiply(const std::vector<double>& a, const std::vector<double>& b, int size);

class StrassenSequential : public ppc::core::Task {
 public:
  explicit StrassenSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  static std::vector<double> AddMatrices(const std::vector<double>& a, const std::vector<double>& b, int size);
  static std::vector<double> SubtractMatrices(const std::vector<double>& a, const std::vector<double>& b, int size);
  static void SplitMatrix(const std::vector<double>& parent, std::vector<double>& child, int row_start, int col_start,
                          int parent_size);
  static void MergeMatrix(std::vector<double>& parent, const std::vector<double>& child, int row_start, int col_start,
                          int parent_size);
  static std::vector<double> PadMatrixToPowerOfTwo(const std::vector<double>& matrix, int original_size);
  static std::vector<double> TrimMatrixToOriginalSize(const std::vector<double>& matrix, int original_size,
                                                      int padded_size);
  static std::vector<double> StrassenMultiply(const std::vector<double>& a, const std::vector<double>& b, int size);

  std::vector<double> input_matrix_a_, input_matrix_b_;
  std::vector<double> output_matrix_;
  int matrix_size_{};
  int original_size_{};
};

}  // namespace nasedkin_e_strassen_algorithm_seq