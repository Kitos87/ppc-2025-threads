#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/malyshev_v_radix_sort/include/ops_seq.hpp"

namespace malyshev_v_radix_sort_seq {
namespace {
std::vector<double> GenerateRandomVector(int size, double min_value, double max_value) {
  std::vector<double> random_vector(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(min_value, max_value);
  for (int i = 0; i < size; ++i) {
    random_vector[i] = dis(gen);
  }
  return random_vector;
}
}  // namespace
}  // namespace malyshev_v_radix_sort_seq

TEST(malyshev_v_radix_sort_seq, ordinary_test) {
  std::vector<double> input_vector = {3.4, 1.2, 0.5, 7.8, 2.3, 4.5, 6.7, 8.9, 1.0, 0.2, 5.6, 4.3, 9.1, 1.5, 3.0};
  std::vector<double> out(15, 0.0);
  std::vector<double> sorted_vector = {0.2, 0.5, 1.0, 1.2, 1.5, 2.3, 3.0, 3.4, 4.3, 4.5, 5.6, 6.7, 7.8, 8.9, 9.1};

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data_seq->inputs_count.emplace_back(input_vector.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  malyshev_v_radix_sort_seq::RadixSortSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  ASSERT_EQ(out, sorted_vector);
}

TEST(malyshev_v_radix_sort_seq, random_vector_test) {
  std::vector<double> input_vector = malyshev_v_radix_sort_seq::GenerateRandomVector(10, -100.0, 100.0);
  std::vector<double> out(10, 0.0);
  std::vector<double> reference = input_vector;
  std::ranges::sort(reference);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data_seq->inputs_count.emplace_back(input_vector.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  malyshev_v_radix_sort_seq::RadixSortSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  ASSERT_EQ(out, reference);
}

TEST(malyshev_v_radix_sort_seq, negative_numbers_test) {
  std::vector<double> input_vector = {-5.4, -2.3, -9.1, -0.5, -3.7, -1.2};
  std::vector<double> out(6, 0.0);
  std::vector<double> sorted_vector = {-9.1, -5.4, -3.7, -2.3, -1.2, -0.5};

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data_seq->inputs_count.emplace_back(input_vector.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  malyshev_v_radix_sort_seq::RadixSortSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  ASSERT_EQ(out, sorted_vector);
}

TEST(malyshev_v_radix_sort_seq, zeros_test) {
  std::vector<double> input_vector = {0.0, -0.0, 0.0, 0.0, -0.0};
  std::vector<double> out(5, 0.0);
  std::vector<double> sorted_vector = {-0.0, -0.0, 0.0, 0.0, 0.0};

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data_seq->inputs_count.emplace_back(input_vector.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  malyshev_v_radix_sort_seq::RadixSortSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  ASSERT_EQ(out, sorted_vector);
}

TEST(malyshev_v_radix_sort_seq, duplicates_test) {
  std::vector<double> input_vector = {3.3, 1.1, 2.2, 3.3, 1.1, 2.2};
  std::vector<double> out(6, 0.0);
  std::vector<double> sorted_vector = {1.1, 1.1, 2.2, 2.2, 3.3, 3.3};

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data_seq->inputs_count.emplace_back(input_vector.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  malyshev_v_radix_sort_seq::RadixSortSequential test_task_sequential(task_data_seq);
  ASSERT_TRUE(test_task_sequential.Validation());
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  ASSERT_EQ(out, sorted_vector);

  for (size_t i = 1; i < out.size(); ++i) {
    ASSERT_LE(out[i - 1], out[i]);
  }
}

TEST(malyshev_v_radix_sort_seq, Validation_test) {
  std::vector<double> input_vector;
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data_seq->inputs_count.emplace_back(input_vector.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  malyshev_v_radix_sort_seq::RadixSortSequential test_task_sequential(task_data_seq);
  ASSERT_FALSE(test_task_sequential.Validation());
}