#ifndef INTEGRATE_SEQ_HPP
#define INTEGRATE_SEQ_HPP

#include <memory>
#include <utility>

#include "../include/integrator.hpp"
#include "core/task/include/task.hpp"

namespace khasanyanov_k_trapezoid_method_omp {

struct TaskContext {
  IntegrationFunction function;
  IntegrationBounds bounds;
  double precision;
};

class TrapezoidalMethodOpenMP : public ppc::core::Task {
 public:
  explicit TrapezoidalMethodOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void CreateTaskData(std::shared_ptr<ppc::core::TaskData> &, TaskContext &context, double *);

 private:
  TaskContext data_;
  double res_{};
};

}  // namespace khasanyanov_k_trapezoid_method_omp

#endif