#pragma once

#include <complex>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace solovev_a_matrix {
struct MatrixInCcsSparse {
  std::vector<std::complex<double>> val;
  std::vector<int> row;
  std::vector<int> col_p;

  int r_n;
  int c_n;
  int n_z;

  MatrixInCcsSparse(int r_nn = 0, int c_nn = 0, int n_zz = 0) {
    c_n = c_nn;
    r_n = r_nn;
    n_z = n_zz;
    row.resize(n_z);
    col_p.resize(r_n + 1);
    val.resize(n_z);
  }
};

class SeqMatMultCcs : public ppc::core::Task {
 public:
  explicit SeqMatMultCcs(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  MatrixInCcsSparse *M1_, *M2_, *M3_;
};
}  // namespace solovev_a_matrix
