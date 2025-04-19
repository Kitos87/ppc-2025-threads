#pragma once

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kholin_k_multidimensional_integrals_rectangle_omp {
using Function = std::function<double(const std::vector<double>&)>;
class TestTaskOpenMP : public ppc::core::Task {
 public:
  explicit TestTaskOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> f_values_;
  Function f_;
  std::vector<double> lower_limits_;
  std::vector<double> upper_limits_;
  double start_n_;
  double result_;

  size_t dim_;
  size_t sz_values_;
  size_t sz_lower_limits_;
  size_t sz_upper_limits_;

  double Integrate(const Function& f, const std::vector<double>& l_limits, const std::vector<double>& u_limits,
                   const std::vector<double>& h, std::vector<double> f_values, int curr_index_dim, size_t dim,
                   double n);
  double IntegrateWithRectangleMethod(const Function& f, std::vector<double>& f_values,
                                      const std::vector<double>& l_limits, const std::vector<double>& u_limits,
                                      size_t dim, double n, std::vector<double> h);
  double RunMultistepSchemeMethodRectangle(const Function& f, std::vector<double> f_values,
                                           const std::vector<double>& l_limits, const std::vector<double>& u_limits,
                                           size_t dim, double n);
};

}  // namespace kholin_k_multidimensional_integrals_rectangle_omp