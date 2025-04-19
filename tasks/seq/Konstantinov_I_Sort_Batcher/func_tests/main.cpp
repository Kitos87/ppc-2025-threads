#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/Konstantinov_I_Sort_Batcher/include/ops_seq.hpp"

TEST(Konstantinov_I_Sort_Batcher_seq, test_wrong_size) {
  std::vector<double> in(2, 0.0);
  std::vector<double> out(1);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  konstantinov_i_sort_batcher_seq::RadixSortBatcherSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), false);
}

TEST(Konstantinov_I_Sort_Batcher_seq, test_scalar) {
  std::vector<double> in{3.14};
  std::vector<double> exp_out{3.14};
  std::vector<double> out(1);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  konstantinov_i_sort_batcher_seq::RadixSortBatcherSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  EXPECT_EQ(exp_out, out);
}

TEST(Konstantinov_I_Sort_Batcher_seq, test_negative_values) {
  std::vector<double> in{-3.14, -1.0, -100.5, -0.1, -999.99};
  std::vector<double> exp_out{-999.99, -100.5, -3.14, -1.0, -0.1};
  std::vector<double> out(5);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  konstantinov_i_sort_batcher_seq::RadixSortBatcherSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  EXPECT_EQ(exp_out, out);
}

TEST(Konstantinov_I_Sort_Batcher_seq, test_reverse_sorted) {
  std::vector<double> in = {100.0, 50.0, 1.0, 0.0, -1.0, -50.0, -100.0};
  std::vector<double> exp_out = in;
  std::ranges::sort(exp_out);
  std::vector<double> out(in.size());

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_omp->inputs_count.emplace_back(in.size());
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_omp->outputs_count.emplace_back(out.size());

  konstantinov_i_sort_batcher_seq::RadixSortBatcherSeq test_task_omp(task_data_omp);
  ASSERT_EQ(test_task_omp.ValidationImpl(), true);
  test_task_omp.PreProcessingImpl();
  test_task_omp.RunImpl();
  test_task_omp.PostProcessingImpl();
  EXPECT_EQ(exp_out, out);
}

TEST(Konstantinov_I_Sort_Batcher_seq, test_random_100_values) {
  constexpr size_t kCount = 100;
  std::vector<double> in(kCount);
  std::vector<double> out(kCount);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-1000.0, 1000.0);

  for (auto &num : in) {
    num = dist(gen);
  }
  std::vector<double> exp_out = in;
  std::ranges::sort(exp_out);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  konstantinov_i_sort_batcher_seq::RadixSortBatcherSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  EXPECT_EQ(exp_out, out);
}

TEST(Konstantinov_I_Sort_Batcher_seq, test_random_1000_values) {
  constexpr size_t kCount = 1000;
  std::vector<double> in(kCount);
  std::vector<double> out(kCount);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-10000.0, 10000.0);

  for (auto &num : in) {
    num = dist(gen);
  }
  std::vector<double> exp_out = in;
  std::ranges::sort(exp_out);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  konstantinov_i_sort_batcher_seq::RadixSortBatcherSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  EXPECT_EQ(exp_out, out);
}

TEST(Konstantinov_I_Sort_Batcher_seq, test_random_100000_values) {
  constexpr size_t kCount = 100000;
  std::vector<double> in(kCount);
  std::vector<double> out(kCount);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-10000.0, 10000.0);

  for (auto &num : in) {
    num = dist(gen);
  }
  std::vector<double> exp_out = in;
  std::ranges::sort(exp_out);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  konstantinov_i_sort_batcher_seq::RadixSortBatcherSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  EXPECT_EQ(exp_out, out);
}

TEST(Konstantinov_I_Sort_Batcher_seq, test_random_1000000_values) {
  constexpr size_t kCount = 1000000;
  std::vector<double> in(kCount);
  std::vector<double> out(kCount);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-10000.0, 10000.0);

  for (auto &num : in) {
    num = dist(gen);
  }
  std::vector<double> exp_out = in;
  std::ranges::sort(exp_out);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  konstantinov_i_sort_batcher_seq::RadixSortBatcherSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  EXPECT_EQ(exp_out, out);
}

TEST(Konstantinov_I_Sort_Batcher_seq, test_random_5000000_values) {
  constexpr size_t kCount = 5000000;
  std::vector<double> in(kCount);
  std::vector<double> out(kCount);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-100000.0, 100000.0);

  for (auto &num : in) {
    num = dist(gen);
  }
  std::vector<double> exp_out = in;
  std::ranges::sort(exp_out);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  konstantinov_i_sort_batcher_seq::RadixSortBatcherSeq test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.ValidationImpl(), true);
  test_task_sequential.PreProcessingImpl();
  test_task_sequential.RunImpl();
  test_task_sequential.PostProcessingImpl();
  EXPECT_EQ(exp_out, out);
}