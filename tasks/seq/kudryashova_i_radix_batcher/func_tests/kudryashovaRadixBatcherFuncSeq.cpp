#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <ctime>
#include <functional>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/kudryashova_i_radix_batcher/include/kudryashovaRadixBatcherSeq.hpp"

std::vector<double> kudryashova_i_radix_batcher_seq::GetRandomDoubleVector(int size) {
  std::vector<double> vector(size);
  std::mt19937 generator(static_cast<unsigned>(std::time(nullptr)));
  std::uniform_real_distribution<double> distribution(-1000.0, 1000.0);
  for (int i = 0; i < size; ++i) {
    vector[i] = distribution(generator);
  }
  return vector;
}

TEST(kudryashova_i_radix_batcher_seq, seq_radix_test_0) {
  int global_vector_size = 3;
  std::vector<double> global_vector = {5.69, -2.11, 0.52};
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  task_data->inputs_count.emplace_back(global_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data->outputs_count.emplace_back(result.size());
  kudryashova_i_radix_batcher_seq::TestTaskSequential test_task_sequential(task_data);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  std::vector<double> sorted_global_vector = global_vector;
  std::ranges::sort(sorted_global_vector);
  ASSERT_EQ(result, sorted_global_vector);
}

TEST(kudryashova_i_radix_batcher_seq, seq_radix_test_1) {
  int global_vector_size = 9;
  std::vector<double> global_vector = {8, -2, 5, 10, 1, -7, 3, 12, -6};
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  task_data->inputs_count.emplace_back(global_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data->outputs_count.emplace_back(result.size());
  kudryashova_i_radix_batcher_seq::TestTaskSequential test_task_sequential(task_data);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  std::vector<double> sorted_global_vector = global_vector;
  std::ranges::sort(sorted_global_vector);
  ASSERT_EQ(result, sorted_global_vector);
}

TEST(kudryashova_i_radix_batcher_seq, seq_radix_test_2) {
  int global_vector_size = 11;
  std::vector<double> global_vector = {-8.55,   1.85,   -4.0,   2.81828, 8.77,   -5.56562,
                                       -15.823, -6.971, 3.1615, 0.0,     10.1415};
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  task_data->inputs_count.emplace_back(global_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data->outputs_count.emplace_back(result.size());
  kudryashova_i_radix_batcher_seq::TestTaskSequential test_task_sequential(task_data);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  std::vector<double> sorted_global_vector = global_vector;
  std::ranges::sort(sorted_global_vector);
  ASSERT_EQ(result, sorted_global_vector);
}

TEST(kudryashova_i_radix_batcher_seq, seq_radix_empty_test) {
  int global_vector_size = 0;
  std::vector<double> global_vector;
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  task_data->inputs_count.emplace_back(global_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data->outputs_count.emplace_back(result.size());
  kudryashova_i_radix_batcher_seq::TestTaskSequential test_task_sequential(task_data);
  ASSERT_FALSE(test_task_sequential.ValidationImpl());
}

TEST(kudryashova_i_radix_batcher_seq, seq_radix_random_test_1) {
  int global_vector_size = 10;
  std::vector<double> global_vector = kudryashova_i_radix_batcher_seq::GetRandomDoubleVector(global_vector_size);
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  task_data->inputs_count.emplace_back(global_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data->outputs_count.emplace_back(result.size());
  kudryashova_i_radix_batcher_seq::TestTaskSequential test_task_sequential(task_data);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  std::vector<double> sorted_global_vector = global_vector;
  std::ranges::sort(sorted_global_vector);
  ASSERT_EQ(result, sorted_global_vector);
}

TEST(kudryashova_i_radix_batcher_seq, seq_radix_random_test_2) {
  int global_vector_size = 50;
  std::vector<double> global_vector = kudryashova_i_radix_batcher_seq::GetRandomDoubleVector(global_vector_size);
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  task_data->inputs_count.emplace_back(global_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data->outputs_count.emplace_back(result.size());
  kudryashova_i_radix_batcher_seq::TestTaskSequential test_task_sequential(task_data);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  std::vector<double> sorted_global_vector = global_vector;
  std::ranges::sort(sorted_global_vector);
  ASSERT_EQ(result, sorted_global_vector);
}

TEST(kudryashova_i_radix_batcher_seq, seq_radix_random_test_3) {
  int global_vector_size = 512;
  std::vector<double> global_vector = kudryashova_i_radix_batcher_seq::GetRandomDoubleVector(global_vector_size);
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  task_data->inputs_count.emplace_back(global_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data->outputs_count.emplace_back(result.size());
  kudryashova_i_radix_batcher_seq::TestTaskSequential test_task_sequential(task_data);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  std::vector<double> sorted_global_vector = global_vector;
  std::ranges::sort(sorted_global_vector);
  ASSERT_EQ(result, sorted_global_vector);
}

TEST(kudryashova_i_radix_batcher_seq, seq_radix_random_reverse_order_test_4) {
  int global_vector_size = 123;
  std::vector<double> global_vector = kudryashova_i_radix_batcher_seq::GetRandomDoubleVector(global_vector_size);
  std::ranges::sort(global_vector, std::greater<>());
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  task_data->inputs_count.emplace_back(global_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data->outputs_count.emplace_back(result.size());
  kudryashova_i_radix_batcher_seq::TestTaskSequential test_task_sequential(task_data);
  ASSERT_TRUE(test_task_sequential.ValidationImpl());
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  std::vector<double> sorted_global_vector = global_vector;
  std::ranges::sort(sorted_global_vector);
  ASSERT_EQ(result, sorted_global_vector);
}