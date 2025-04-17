#include "seq/sidorina_p_gradient_method/include/ops_seq.hpp"

bool sidorina_p_gradient_method_seq::GradientMethod::PreProcessingImpl() {
  size_ = *reinterpret_cast<int*>(task_data->inputs[0]);
  tolerance_ = *reinterpret_cast<double*>(task_data->inputs[1]);
  auto* a_ptr = reinterpret_cast<double*>(task_data->inputs[2]);
  unsigned int a_size = task_data->inputs_count[2];
  a_.assign(a_ptr, a_ptr + a_size);
  auto* b_ptr = reinterpret_cast<double*>(task_data->inputs[3]);
  unsigned int b_size = task_data->inputs_count[3];
  b_.assign(b_ptr, b_ptr + b_size);
  auto* solution_ptr = reinterpret_cast<double*>(task_data->inputs[4]);
  unsigned int solution_size = task_data->inputs_count[4];
  solution_.assign(solution_ptr, solution_ptr + solution_size);
  result_.resize(size_);
  return true;
}

bool sidorina_p_gradient_method_seq::GradientMethod::ValidationImpl() {
  if (*reinterpret_cast<int*>(task_data->inputs[0]) <= 0 || static_cast<int>(task_data->inputs_count[2]) <= 0 ||
      static_cast<int>(task_data->inputs_count[3]) <= 0 || static_cast<int>(task_data->inputs_count[4]) <= 0) {
    return false;
  }

  if (*reinterpret_cast<int*>(task_data->inputs[0]) != static_cast<int>(task_data->inputs_count[3]) ||
      *reinterpret_cast<int*>(task_data->inputs[0]) != static_cast<int>(task_data->inputs_count[4]) ||
      *reinterpret_cast<int*>(task_data->inputs[0]) * *reinterpret_cast<int*>(task_data->inputs[0]) !=
          static_cast<int>(task_data->inputs_count[2])) {
    return false;
  }

  if (task_data->inputs_count.size() < 5 || task_data->inputs.size() < 5 || task_data->outputs.empty()) {
    return false;
  }

  if (static_cast<int>(task_data->inputs_count[2]) !=
      (static_cast<int>(task_data->inputs_count[3]) * static_cast<int>(task_data->inputs_count[3]))) {
    return false;
  }

  if (task_data->outputs_count[0] != task_data->inputs_count[4]) {
    return false;
  }

  const auto* matrix = reinterpret_cast<const double*>(task_data->inputs[2]);

  return MatrixSimmPositive(matrix, *reinterpret_cast<int*>(task_data->inputs[0]));
}

bool sidorina_p_gradient_method_seq::GradientMethod::RunImpl() {
  result_ = ConjugateGradientMethod(a_, b_, solution_, tolerance_, size_);
  return true;
}

bool sidorina_p_gradient_method_seq::GradientMethod::PostProcessingImpl() {
  auto* result_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  for (unsigned long i = 0; i < result_.size(); i++) {
    result_ptr[i] = result_[i];
  }
  return true;
}