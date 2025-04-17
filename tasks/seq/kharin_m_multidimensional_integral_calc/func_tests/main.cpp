#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/kharin_m_multidimensional_integral_calc/include/ops_seq.hpp"

// Тест для 2D интеграла (3x3 сетка)
TEST(kharin_m_multidimensional_integral_calc_seq, test_integral_3x3) {
  std::vector<double> in = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  std::vector<size_t> grid_sizes = {3, 3};
  std::vector<double> step_sizes = {1.0, 1.0};
  std::vector<double> out(1, 0.0);
  double expected_out = 45.0;  // Сумма 1+2+...+9 = 45 * 1.0 * 1.0

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(grid_sizes.data()));
  task_data_seq->inputs_count.emplace_back(grid_sizes.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(step_sizes.data()));
  task_data_seq->inputs_count.emplace_back(step_sizes.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kharin_m_multidimensional_integral_calc_seq::TaskSequential task(task_data_seq);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  EXPECT_DOUBLE_EQ(out[0], expected_out);
}

// Тест для 1D интеграла
TEST(kharin_m_multidimensional_integral_calc_seq, test_integral_1d) {
  std::vector<double> in = {1.0, 2.0, 3.0};
  std::vector<size_t> grid_sizes = {3};
  std::vector<double> step_sizes = {0.5};
  std::vector<double> out(1, 0.0);
  double expected_out = (1.0 + 2.0 + 3.0) * 0.5;  // 6.0 * 0.5 = 3.0

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(grid_sizes.data()));
  task_data_seq->inputs_count.emplace_back(grid_sizes.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(step_sizes.data()));
  task_data_seq->inputs_count.emplace_back(step_sizes.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kharin_m_multidimensional_integral_calc_seq::TaskSequential task(task_data_seq);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  EXPECT_DOUBLE_EQ(out[0], expected_out);
}

// Тест для 3D интеграла (2x2x2 сетка)
TEST(kharin_m_multidimensional_integral_calc_seq, test_integral_3d) {
  std::vector<double> in(8, 1.0);  // 2x2x2 сетка, все значения = 1.0
  std::vector<size_t> grid_sizes = {2, 2, 2};
  std::vector<double> step_sizes = {1.0, 1.0, 1.0};
  std::vector<double> out(1, 0.0);
  double expected_out = 8.0 * 1.0 * 1.0 * 1.0;  // 8.0

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(grid_sizes.data()));
  task_data_seq->inputs_count.emplace_back(grid_sizes.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(step_sizes.data()));
  task_data_seq->inputs_count.emplace_back(step_sizes.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kharin_m_multidimensional_integral_calc_seq::TaskSequential task(task_data_seq);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  EXPECT_DOUBLE_EQ(out[0], expected_out);
}

TEST(kharin_m_multidimensional_integral_calc_seq, test_invalid_input_count) {
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  // Добавляем только два входа вместо трех
  task_data_seq->inputs.emplace_back(nullptr);
  task_data_seq->inputs_count.emplace_back(0);
  task_data_seq->inputs.emplace_back(nullptr);
  task_data_seq->inputs_count.emplace_back(0);
  task_data_seq->outputs.emplace_back(nullptr);
  task_data_seq->outputs_count.emplace_back(1);

  kharin_m_multidimensional_integral_calc_seq::TaskSequential task(task_data_seq);
  EXPECT_FALSE(task.Validation());
}

TEST(kharin_m_multidimensional_integral_calc_seq, test_invalid_output_count) {
  std::vector<double> in = {1.0, 2.0, 3.0};
  std::vector<size_t> grid_sizes = {3};
  std::vector<double> step_sizes = {0.5};
  std::vector<double> out(2, 0.0);  // Некорректно: должно быть 1 выходное значение

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(grid_sizes.data()));
  task_data_seq->inputs_count.emplace_back(grid_sizes.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(step_sizes.data()));
  task_data_seq->inputs_count.emplace_back(step_sizes.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());  // 2 вместо 1

  kharin_m_multidimensional_integral_calc_seq::TaskSequential task(task_data_seq);
  EXPECT_FALSE(task.Validation());
}

TEST(kharin_m_multidimensional_integral_calc_seq, test_mismatch_grid_step_sizes) {
  std::vector<double> in = {1.0, 2.0, 3.0};
  std::vector<size_t> grid_sizes = {3};         // 1 измерение
  std::vector<double> step_sizes = {0.5, 0.5};  // 2 шага — несоответствие
  std::vector<double> out(1, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(grid_sizes.data()));
  task_data_seq->inputs_count.emplace_back(grid_sizes.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(step_sizes.data()));
  task_data_seq->inputs_count.emplace_back(step_sizes.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kharin_m_multidimensional_integral_calc_seq::TaskSequential task(task_data_seq);
  EXPECT_FALSE(task.Validation());
}

TEST(kharin_m_multidimensional_integral_calc_seq, test_invalid_input_size) {
  std::vector<double> in = {1.0, 2.0};  // 2 элемента, но для сетки 3x3 нужно 9
  std::vector<size_t> grid_sizes = {3, 3};
  std::vector<double> step_sizes = {1.0, 1.0};
  std::vector<double> out(1, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(grid_sizes.data()));
  task_data_seq->inputs_count.emplace_back(grid_sizes.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(step_sizes.data()));
  task_data_seq->inputs_count.emplace_back(step_sizes.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kharin_m_multidimensional_integral_calc_seq::TaskSequential task(task_data_seq);
  ASSERT_TRUE(task.Validation());  // Валидация должна пройти, так как размеры входов проверяются в PreProcessing
  EXPECT_FALSE(task.PreProcessing());
}

TEST(kharin_m_multidimensional_integral_calc_seq, test_random_data) {
  constexpr size_t kDim = 100;
  std::vector<double> in(kDim * kDim);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  double expected_sum = 0.0;
  for (auto& val : in) {
    val = dis(gen);
    expected_sum += val;
  }
  std::vector<size_t> grid_sizes = {kDim, kDim};
  std::vector<double> step_sizes = {0.1, 0.1};
  std::vector<double> out(1, 0.0);
  double expected_out = expected_sum * 0.1 * 0.1;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(grid_sizes.data()));
  task_data_seq->inputs_count.emplace_back(grid_sizes.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(step_sizes.data()));
  task_data_seq->inputs_count.emplace_back(step_sizes.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kharin_m_multidimensional_integral_calc_seq::TaskSequential task(task_data_seq);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  EXPECT_NEAR(out[0], expected_out, 1e-6);
}

TEST(kharin_m_multidimensional_integral_calc_seq, test_negative_step) {
  std::vector<double> in = {1.0, 2.0, 3.0};
  std::vector<size_t> grid_sizes = {3};
  std::vector<double> step_sizes = {-0.5};  // Отрицательный шаг
  std::vector<double> out(1, 0.0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(grid_sizes.data()));
  task_data_seq->inputs_count.emplace_back(grid_sizes.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(step_sizes.data()));
  task_data_seq->inputs_count.emplace_back(step_sizes.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kharin_m_multidimensional_integral_calc_seq::TaskSequential task(task_data_seq);
  ASSERT_TRUE(task.Validation());      // Валидация структуры входов проходит
  EXPECT_FALSE(task.PreProcessing());  // Предобработка должна завершиться неудачно из-за отрицательного шага
}

TEST(kharin_m_multidimensional_integral_calc_seq, test_integral_4d) {
  std::vector<double> in(16, 1.0);  // 2x2x2x2 сетка, все значения = 1.0
  std::vector<size_t> grid_sizes = {2, 2, 2, 2};
  std::vector<double> step_sizes = {1.0, 1.0, 1.0, 1.0};
  std::vector<double> out(1, 0.0);
  double expected_out = 16.0 * 1.0 * 1.0 * 1.0 * 1.0;  // 16 точек * объем 1.0 = 16.0

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(grid_sizes.data()));
  task_data_seq->inputs_count.emplace_back(grid_sizes.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(step_sizes.data()));
  task_data_seq->inputs_count.emplace_back(step_sizes.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kharin_m_multidimensional_integral_calc_seq::TaskSequential task(task_data_seq);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  EXPECT_DOUBLE_EQ(out[0], expected_out);
}

TEST(kharin_m_multidimensional_integral_calc_seq, test_integral_5d) {
  std::vector<double> in(3 * 2 * 2 * 2 * 2, 2.0);  // 3x2x2x2x2 сетка, все значения = 2.0
  std::vector<size_t> grid_sizes = {3, 2, 2, 2, 2};
  std::vector<double> step_sizes = {0.5, 0.5, 0.5, 0.5, 0.5};
  std::vector<double> out(1, 0.0);
  size_t total_points = 3 * 2 * 2 * 2 * 2;
  double sum_values = static_cast<double>(total_points) * 2.0;
  double volume_element = 0.5 * 0.5 * 0.5 * 0.5 * 0.5;
  double expected_out = sum_values * volume_element;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(grid_sizes.data()));
  task_data_seq->inputs_count.emplace_back(grid_sizes.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(step_sizes.data()));
  task_data_seq->inputs_count.emplace_back(step_sizes.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kharin_m_multidimensional_integral_calc_seq::TaskSequential task(task_data_seq);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  EXPECT_DOUBLE_EQ(out[0], expected_out);
}