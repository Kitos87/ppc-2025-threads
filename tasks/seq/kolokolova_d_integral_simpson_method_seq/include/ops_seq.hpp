#pragma once

#include <functional>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kolokolova_d_integral_simpson_method_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data, std::function<double(std::vector<double>)> func)
      : Task(std::move(task_data)), func_(std::move(func)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  std::vector<double> FindFunctionValue(const std::vector<std::vector<double>>& coordinates,
                                        const std::function<double(std::vector<double>)>& f);
  void GeneratePointsAndEvaluate(const std::vector<std::vector<double>>& coordinates, int index,
                                 std::vector<double>& current, std::vector<double>& results,
                                 const std::function<double(const std::vector<double>)>& f);
  static std::vector<double> FindCoeff(int count_step);
  static void MultiplyCoeffandFunctionValue(std::vector<double>& function_val, const std::vector<double>& coeff_vec,
                                            int a);
  [[nodiscard]] double CreateOutputResult(std::vector<double> vec, std::vector<double> size_steps) const;
  static bool CheckBorders(std::vector<int> vec);

 private:
  double result_output_ = 0;
  int nums_variables_ = 0;
  std::vector<int> steps_;
  std::vector<int> borders_;
  std::function<double(std::vector<double>)> func_;
};

}  // namespace kolokolova_d_integral_simpson_method_seq