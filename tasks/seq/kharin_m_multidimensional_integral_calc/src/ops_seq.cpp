#include "seq/kharin_m_multidimensional_integral_calc/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

bool kharin_m_multidimensional_integral_calc_seq::TaskSequential::ValidationImpl() {
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

bool kharin_m_multidimensional_integral_calc_seq::TaskSequential::PreProcessingImpl() {
  size_t d = task_data->inputs_count[1];
  auto* sizes_ptr = reinterpret_cast<size_t*>(task_data->inputs[1]);
  grid_sizes_ = std::vector<size_t>(sizes_ptr, sizes_ptr + d);

  // Вычисляем общее количество точек сетки
  size_t total_size = 1;
  for (const auto& n : grid_sizes_) {
    total_size *= n;
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

  // Проверка на отрицательные шаги
  for (const auto& h : step_sizes_) {
    if (h <= 0.0) {
      return false;  // Отрицательный или нулевой шаг недопустим
    }
  }

  output_result_ = 0.0;
  return true;
}

bool kharin_m_multidimensional_integral_calc_seq::TaskSequential::RunImpl() {
  // Вычисляем сумму всех значений функции
  double total = 0.0;
  for (const auto& val : input_) {
    total += val;
  }
  // Вычисляем элемент объема как произведение шагов интегрирования
  double volume_element = 1.0;
  for (const auto& h : step_sizes_) {
    volume_element *= h;
  }
  // Интеграл = сумма значений * элемент объема
  output_result_ = total * volume_element;
  return true;
}

bool kharin_m_multidimensional_integral_calc_seq::TaskSequential::PostProcessingImpl() {
  reinterpret_cast<double*>(task_data->outputs[0])[0] = output_result_;
  return true;
}