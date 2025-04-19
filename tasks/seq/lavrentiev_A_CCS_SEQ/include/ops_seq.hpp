#pragma once

#include <cmath>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace lavrentiev_a_ccs_seq {

struct Sparse {
  std::pair<int, int> size;
  std::vector<double> elements;
  std::vector<int> rows;
  std::vector<int> columnsSum;
};

class CCSSequential : public ppc::core::Task {
 private:
  [[nodiscard]] bool IsEmpty() const;
  static Sparse ConvertToSparse(std::pair<int, int> bsize, const std::vector<double>& values);
  static Sparse Transpose(const Sparse& sparse);
  static Sparse MatMul(const Sparse& matrix1, const Sparse& matrix2);
  static int GetElementsCount(int index, const std::vector<int>& columns_sum);
  static std::vector<double> ConvertFromSparse(const Sparse& matrix);

  Sparse A_;
  Sparse B_;
  Sparse Answer_;

 public:
  explicit CCSSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace lavrentiev_a_ccs_seq