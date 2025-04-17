#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/deryabin_m_hoare_sort_simple_merge/include/ops_seq.hpp"

TEST(deryabin_m_hoare_sort_simple_merge_seq, test_short_array) {
  // Create data
  std::vector<double> input_array{-1, -2, -3, -11, -22, -33};
  std::vector<std::vector<double>> in_array(1, input_array);
  size_t chunk_count = 2;
  std::vector<double> output_array(6);
  std::vector<std::vector<double>> out_array(1, output_array);
  std::vector<double> true_solution{-33, -22, -11, -3, -2, -1};

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_array.data()));
  task_data_seq->inputs_count.emplace_back(input_array.size());
  task_data_seq->inputs_count.emplace_back(chunk_count);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_array.data()));
  task_data_seq->outputs_count.emplace_back(output_array.size());

  // Create Task
  deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential hoare_sort_task_sequential(task_data_seq);
  ASSERT_EQ(hoare_sort_task_sequential.Validation(), true);
  hoare_sort_task_sequential.PreProcessing();
  hoare_sort_task_sequential.Run();
  hoare_sort_task_sequential.PostProcessing();
  ASSERT_EQ(true_solution, out_array[0]);
}

TEST(deryabin_m_hoare_sort_simple_merge_seq, test_random_array) {
  // Create data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distribution(-100, 100);
  std::vector<double> input_array(800);
  std::ranges::generate(input_array.begin(), input_array.end(), [&] { return distribution(gen); });
  std::vector<std::vector<double>> in_array(1, input_array);
  size_t chunk_count = 8;
  std::vector<double> output_array(800);
  std::vector<std::vector<double>> out_array(1, output_array);
  std::vector<double> true_solution(input_array);
  std::ranges::sort(true_solution.begin(), true_solution.end());

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_array.data()));
  task_data_seq->inputs_count.emplace_back(input_array.size());
  task_data_seq->inputs_count.emplace_back(chunk_count);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_array.data()));
  task_data_seq->outputs_count.emplace_back(output_array.size());

  // Create Task
  deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential hoare_sort_task_sequential(task_data_seq);
  ASSERT_EQ(hoare_sort_task_sequential.Validation(), true);
  hoare_sort_task_sequential.PreProcessing();
  hoare_sort_task_sequential.Run();
  hoare_sort_task_sequential.PostProcessing();
  ASSERT_EQ(true_solution, out_array[0]);
}

TEST(deryabin_m_hoare_sort_simple_merge_seq, test_negative_elements_array) {
  // Create data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distribution(-100, -1);
  std::vector<double> input_array(800);
  std::ranges::generate(input_array.begin(), input_array.end(), [&] { return distribution(gen); });
  std::vector<std::vector<double>> in_array(1, input_array);
  size_t chunk_count = 8;
  std::vector<double> output_array(800);
  std::vector<std::vector<double>> out_array(1, output_array);
  std::vector<double> true_solution(input_array);
  std::ranges::sort(true_solution.begin(), true_solution.end());

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_array.data()));
  task_data_seq->inputs_count.emplace_back(input_array.size());
  task_data_seq->inputs_count.emplace_back(chunk_count);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_array.data()));
  task_data_seq->outputs_count.emplace_back(output_array.size());

  // Create Task
  deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential hoare_sort_task_sequential(task_data_seq);
  ASSERT_EQ(hoare_sort_task_sequential.Validation(), true);
  hoare_sort_task_sequential.PreProcessing();
  hoare_sort_task_sequential.Run();
  hoare_sort_task_sequential.PostProcessing();
  ASSERT_EQ(true_solution, out_array[0]);
}

TEST(deryabin_m_hoare_sort_simple_merge_seq, test_shuffle_array) {
  // Create data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<double> input_array(800);
  std::ranges::generate(input_array.begin(), input_array.end(), [value = 0]() mutable { return value++; });
  std::shuffle(input_array.begin(), input_array.end(), gen);
  std::vector<std::vector<double>> in_array(1, input_array);
  size_t chunk_count = 8;
  std::vector<double> output_array(800);
  std::vector<std::vector<double>> out_array(1, output_array);
  std::vector<double> true_solution(input_array);
  std::ranges::sort(true_solution.begin(), true_solution.end());

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_array.data()));
  task_data_seq->inputs_count.emplace_back(input_array.size());
  task_data_seq->inputs_count.emplace_back(chunk_count);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_array.data()));
  task_data_seq->outputs_count.emplace_back(output_array.size());

  // Create Task
  deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential hoare_sort_task_sequential(task_data_seq);
  ASSERT_EQ(hoare_sort_task_sequential.Validation(), true);
  hoare_sort_task_sequential.PreProcessing();
  hoare_sort_task_sequential.Run();
  hoare_sort_task_sequential.PostProcessing();
  ASSERT_EQ(true_solution, out_array[0]);
}

TEST(deryabin_m_hoare_sort_simple_merge_seq, test_random_array_small_pieces) {
  // Create data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distribution(-100, 100);
  std::vector<double> input_array(800);
  std::ranges::generate(input_array.begin(), input_array.end(), [&] { return distribution(gen); });
  std::vector<std::vector<double>> in_array(1, input_array);
  size_t chunk_count = 32;
  std::vector<double> output_array(800);
  std::vector<std::vector<double>> out_array(1, output_array);
  std::vector<double> true_solution(input_array);
  std::ranges::sort(true_solution.begin(), true_solution.end());

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_array.data()));
  task_data_seq->inputs_count.emplace_back(input_array.size());
  task_data_seq->inputs_count.emplace_back(chunk_count);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_array.data()));
  task_data_seq->outputs_count.emplace_back(output_array.size());

  // Create Task
  deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential hoare_sort_task_sequential(task_data_seq);
  ASSERT_EQ(hoare_sort_task_sequential.Validation(), true);
  hoare_sort_task_sequential.PreProcessing();
  hoare_sort_task_sequential.Run();
  hoare_sort_task_sequential.PostProcessing();
  ASSERT_EQ(true_solution, out_array[0]);
}

TEST(deryabin_m_hoare_sort_simple_merge_seq, test_random_array_large_pieces) {
  // Create data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distribution(-100, 100);
  std::vector<double> input_array(800);
  std::ranges::generate(input_array.begin(), input_array.end(), [&] { return distribution(gen); });
  std::vector<std::vector<double>> in_array(1, input_array);
  size_t chunk_count = 2;
  std::vector<double> output_array(800);
  std::vector<std::vector<double>> out_array(1, output_array);
  std::vector<double> true_solution(input_array);
  std::ranges::sort(true_solution.begin(), true_solution.end());

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_array.data()));
  task_data_seq->inputs_count.emplace_back(input_array.size());
  task_data_seq->inputs_count.emplace_back(chunk_count);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_array.data()));
  task_data_seq->outputs_count.emplace_back(output_array.size());

  // Create Task
  deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential hoare_sort_task_sequential(task_data_seq);
  ASSERT_EQ(hoare_sort_task_sequential.Validation(), true);
  hoare_sort_task_sequential.PreProcessing();
  hoare_sort_task_sequential.Run();
  hoare_sort_task_sequential.PostProcessing();
  ASSERT_EQ(true_solution, out_array[0]);
}

TEST(deryabin_m_hoare_sort_simple_merge_seq, test_random_array_large_pieces_reversed_order) {
  // Create data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<double> input_array(800);
  std::ranges::generate(input_array.begin(), input_array.end(), [value = 800]() mutable { return value--; });
  std::vector<std::vector<double>> in_array(1, input_array);
  size_t chunk_count = 2;
  std::vector<double> output_array(800);
  std::vector<std::vector<double>> out_array(1, output_array);
  std::vector<double> true_solution(input_array);
  std::ranges::sort(true_solution.begin(), true_solution.end());

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_array.data()));
  task_data_seq->inputs_count.emplace_back(input_array.size());
  task_data_seq->inputs_count.emplace_back(chunk_count);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_array.data()));
  task_data_seq->outputs_count.emplace_back(output_array.size());

  // Create Task
  deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential hoare_sort_task_sequential(task_data_seq);
  ASSERT_EQ(hoare_sort_task_sequential.Validation(), true);
  hoare_sort_task_sequential.PreProcessing();
  hoare_sort_task_sequential.Run();
  hoare_sort_task_sequential.PostProcessing();
  ASSERT_EQ(true_solution, out_array[0]);
}

TEST(deryabin_m_hoare_sort_simple_merge_seq, test_partially_sorted_array) {
  // Create data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<double> input_array(800);
  std::ranges::generate(input_array.begin(), input_array.end(), [value = 0]() mutable { return value++; });
  const auto half = input_array.size() / 2U;
  std::shuffle(input_array.begin() + (long)half, input_array.end(), gen);
  std::vector<std::vector<double>> in_array(1, input_array);
  size_t chunk_count = 8;
  std::vector<double> output_array(800);
  std::vector<std::vector<double>> out_array(1, output_array);
  std::vector<double> true_solution(input_array);
  std::ranges::sort(true_solution.begin(), true_solution.end());

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_array.data()));
  task_data_seq->inputs_count.emplace_back(input_array.size());
  task_data_seq->inputs_count.emplace_back(chunk_count);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_array.data()));
  task_data_seq->outputs_count.emplace_back(output_array.size());

  // Create Task
  deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential hoare_sort_task_sequential(task_data_seq);
  ASSERT_EQ(hoare_sort_task_sequential.Validation(), true);
  hoare_sort_task_sequential.PreProcessing();
  hoare_sort_task_sequential.Run();
  hoare_sort_task_sequential.PostProcessing();
  ASSERT_EQ(true_solution, out_array[0]);
}

TEST(deryabin_m_hoare_sort_simple_merge_seq, test_backward_sorted_array) {
  // Create data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<double> input_array(800);
  std::ranges::generate(input_array.begin(), input_array.end(), [value = 800]() mutable { return value--; });
  std::vector<std::vector<double>> in_array(1, input_array);
  size_t chunk_count = 8;
  std::vector<double> output_array(800);
  std::vector<std::vector<double>> out_array(1, output_array);
  std::vector<double> true_solution(input_array);
  std::ranges::sort(true_solution.begin(), true_solution.end());

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_array.data()));
  task_data_seq->inputs_count.emplace_back(input_array.size());
  task_data_seq->inputs_count.emplace_back(chunk_count);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_array.data()));
  task_data_seq->outputs_count.emplace_back(output_array.size());

  // Create Task
  deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential hoare_sort_task_sequential(task_data_seq);
  ASSERT_EQ(hoare_sort_task_sequential.Validation(), true);
  hoare_sort_task_sequential.PreProcessing();
  hoare_sort_task_sequential.Run();
  hoare_sort_task_sequential.PostProcessing();
  ASSERT_EQ(true_solution, out_array[0]);
}

TEST(deryabin_m_hoare_sort_simple_merge_seq, test_invalid_array) {
  // Create data
  std::vector<double> input_array(2);
  std::vector<std::vector<double>> in_array(1, input_array);
  size_t chunk_count = 2;
  std::vector<double> output_array(2);
  std::vector<std::vector<double>> out_array(1, output_array);

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_array.data()));
  task_data_seq->inputs_count.emplace_back(input_array.size());
  task_data_seq->inputs_count.emplace_back(chunk_count);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_array.data()));
  task_data_seq->outputs_count.emplace_back(output_array.size());

  // Create Task
  deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential hoare_sort_task_sequential(task_data_seq);
  ASSERT_EQ(hoare_sort_task_sequential.Validation(), false);
}

TEST(deryabin_m_hoare_sort_simple_merge_seq, test_invalid_chunk_count) {
  // Create data
  std::vector<double> input_array{-1, -2, -3, -11, -22, -33};
  std::vector<std::vector<double>> in_array(1, input_array);
  size_t chunk_count = 0;
  std::vector<double> output_array(6);
  std::vector<std::vector<double>> out_array(1, output_array);

  // Create TaskData
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_array.data()));
  task_data_seq->inputs_count.emplace_back(input_array.size());
  task_data_seq->inputs_count.emplace_back(chunk_count);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_array.data()));
  task_data_seq->outputs_count.emplace_back(output_array.size());

  // Create Task
  deryabin_m_hoare_sort_simple_merge_seq::HoareSortTaskSequential hoare_sort_task_sequential(task_data_seq);
  ASSERT_EQ(hoare_sort_task_sequential.Validation(), false);
}
