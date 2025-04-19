#include "../include/integrate_seq.hpp"

#include <cstdint>
#include <memory>

#include "core/task/include/task.hpp"
#include "seq/khasanyanov_k_trapezoid_method/include/integrator.hpp"

using namespace khasanyanov_k_trapezoid_method_seq;

void TrapezoidalMethodSequential::CreateTaskData(std::shared_ptr<ppc::core::TaskData> &task_data, TaskContext &context,
                                                 double *out) {
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&context));
  task_data->inputs_count.emplace_back(context.bounds.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out));
  task_data->outputs_count.emplace_back(1);
}

bool TrapezoidalMethodSequential::ValidationImpl() {
  auto *data = reinterpret_cast<TaskContext *>(task_data->inputs[0]);
  return data != nullptr && task_data->inputs_count[0] > 0 && task_data->outputs[0] != nullptr;
}

bool TrapezoidalMethodSequential::PreProcessingImpl() {
  data_ = *reinterpret_cast<TaskContext *>(task_data->inputs[0]);
  return true;
}
bool TrapezoidalMethodSequential::RunImpl() {
  res_ = Integrator<kSequential>{}(data_.function, data_.bounds, data_.precision);
  return true;
}
bool TrapezoidalMethodSequential::PostProcessingImpl() {
  *reinterpret_cast<double *>(task_data->outputs[0]) = res_;
  return true;
}
