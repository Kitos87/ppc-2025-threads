#include "omp/kharin_m_multidimensional_integral_calc/include/ops_omp.hpp"

#include <omp.h>

#include <cstddef>
#include <vector>

bool kharin_m_multidimensional_integral_calc_omp::TestTaskOpenMP::ValidationImpl() {
  // Проверяем, что предоставлено ровно 3 входа и 1 выход
  if (task_data->inputs.size() != 3 || task_data->outputs.size() != 1) {
    return false;
  }
  // Совпадение grid_sizes и step_sizes
  if (task_data->inputs_count[1] != task_data->inputs_count[2]) {
    return false;
  }
  // Выход должен содержать одно значение
  if (task_data->outputs_count[0] != 1) {
    return false;
  }
  return true;
}

bool kharin_m_multidimensional_integral_calc_omp::TestTaskOpenMP::PreProcessingImpl() {
  size_t d = task_data->inputs_count[1];
  auto* sizes_ptr = reinterpret_cast<size_t*>(task_data->inputs[1]);
  grid_sizes_ = std::vector<size_t>(sizes_ptr, sizes_ptr + d);

  // Вычисляем общее количество точек сетки
  size_t total_size = 1;

#pragma omp parallel for reduction(* : total_size)
  for (int i = 0; i < static_cast<int>(grid_sizes_.size()); i++) {
    total_size *= grid_sizes_[i];
  }

  if (task_data->inputs_count[0] != total_size) {
    return false;
  }
  auto* input_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  input_ = std::vector<double>(input_ptr, input_ptr + total_size);

  if (task_data->inputs_count[2] != d) {
    return false;
  }
  auto* steps_ptr = reinterpret_cast<double*>(task_data->inputs[2]);
  step_sizes_ = std::vector<double>(steps_ptr, steps_ptr + d);

  bool is_valid = true;

#pragma omp parallel for reduction(&& : is_valid)
  for (int i = 0; i < static_cast<int>(step_sizes_.size()); i++) {
    if (step_sizes_[i] <= 0.0) {
      is_valid = false;
    }
  }

  if (is_valid) {
    output_result_ = 0.0;
    return true;
  }
  return false;
}

bool kharin_m_multidimensional_integral_calc_omp::TestTaskOpenMP::RunImpl() {
  // Вычисляем сумму всех значений функции
  double total = 0.0;
#pragma omp parallel for reduction(+ : total)
  for (int i = 0; i < static_cast<int>(input_.size()); i++) {
    total += input_[i];
  }

  // Вычисляем элемент объема как произведение шагов интегрирования
  double volume_element = 1.0;
#pragma omp parallel for reduction(* : volume_element)
  for (int i = 0; i < static_cast<int>(step_sizes_.size()); i++) {
    volume_element *= step_sizes_[i];
  }

  output_result_ = total * volume_element;
  return true;
}

bool kharin_m_multidimensional_integral_calc_omp::TestTaskOpenMP::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = output_result_;
  return true;
}