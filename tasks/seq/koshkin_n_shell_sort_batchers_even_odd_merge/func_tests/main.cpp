#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/koshkin_n_shell_sort_batchers_even_odd_merge/include/ops_seq.hpp"

TEST(koshkin_n_shell_sort_batchers_even_odd_merge_seq, doubleRerverseOrderAscending) {
  bool order = true;
  std::vector<int> in = {5, 4, 3, 2, 1, 5, 4, 3, 2, 1};
  std::vector<int> out(in.size(), 0);

  std::vector<int> res = in;
  std::ranges::sort(res.begin(), res.end());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&order));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, res);
}

TEST(koshkin_n_shell_sort_batchers_even_odd_merge_seq, doubleRerverseOrderDescending) {
  bool order = false;
  std::vector<int> in = {5, 4, 3, 2, 1, 5, 4, 3, 2, 1};
  std::vector<int> out(in.size(), 0);

  std::vector<int> res = in;
  std::ranges::sort(res.begin(), res.end(), std::greater<>());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&order));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, res);
}

TEST(koshkin_n_shell_sort_batchers_even_odd_merge_seq, sortedVectorAscending) {
  bool order = true;

  std::vector<int> in = {-5, 0, 0, 1, 5, 77, 1600, 1700, 1900, 9999};
  std::vector<int> out(in.size(), 0);

  std::vector<int> res = in;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&order));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, res);
}

TEST(koshkin_n_shell_sort_batchers_even_odd_merge_seq, emptyVec) {
  bool order = true;

  std::vector<int> in = {};
  std::vector<int> out(in.size(), 0);

  std::vector<int> res = in;
  std::ranges::sort(res.begin(), res.end());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&order));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}

TEST(koshkin_n_shell_sort_batchers_even_odd_merge_seq, posnegVectorAscending) {
  bool order = true;

  std::vector<int> in = {34, 8, -64, 51, 32, -21, 99, 3, 45, 12, 0, 0};
  std::vector<int> out(in.size(), 0);

  std::vector<int> res = in;
  std::ranges::sort(res.begin(), res.end());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&order));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, res);
}

TEST(koshkin_n_shell_sort_batchers_even_odd_merge_seq, posnegVectorDescending) {
  bool order = false;

  std::vector<int> in = {34, 8, -64, 51, 32, -21, 99, 3, 45, 12, 0, 0};
  std::vector<int> out(in.size(), 0);

  std::vector<int> res = in;
  std::ranges::sort(res.begin(), res.end(), std::greater<>());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&order));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, res);
}

TEST(koshkin_n_shell_sort_batchers_even_odd_merge_seq, positiveVectorAscending) {
  bool order = true;

  std::vector<int> in = {34, 8, 64, 51, 32, 21, 99, 3, 45, 12};
  std::vector<int> out(in.size(), 0);

  std::vector<int> res = in;
  std::ranges::sort(res.begin(), res.end());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&order));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, res);
}

TEST(koshkin_n_shell_sort_batchers_even_odd_merge_seq, negativeVectorAscending) {
  bool order = true;

  std::vector<int> in = {-34, -8, -64, -51, -32, -21, -99, -3, -45, -12};
  std::vector<int> out(in.size(), 0);

  std::vector<int> res = in;
  std::ranges::sort(res.begin(), res.end());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&order));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, res);
}

TEST(koshkin_n_shell_sort_batchers_even_odd_merge_seq, negativeVectorDescending) {
  bool order = false;

  std::vector<int> in = {-34, -8, -64, -51, -32, -21, -99, -3, -45, -12};
  std::vector<int> out(in.size(), 0);

  std::vector<int> res = in;
  std::ranges::sort(res.begin(), res.end(), std::greater<>());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&order));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, res);
}

TEST(koshkin_n_shell_sort_batchers_even_odd_merge_seq, positiveVectorDescending) {
  bool order = false;

  std::vector<int> in = {34, 8, 64, 51, 32, 21, 99, 3, 45, 12};
  std::vector<int> out(in.size(), 0);

  std::vector<int> res = in;
  std::ranges::sort(res.begin(), res.end(), std::greater<>());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&order));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, res);
}

TEST(koshkin_n_shell_sort_batchers_even_odd_merge_seq, smallVectorDescending) {
  bool order = false;

  std::vector<int> in = koshkin_n_shell_sort_batchers_even_odd_merge_seq::GetRandomVector(15);
  std::vector<int> out(in.size(), 0);

  std::vector<int> res = in;
  std::ranges::sort(res.begin(), res.end(), std::greater<>());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&order));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, res);
}

TEST(koshkin_n_shell_sort_batchers_even_odd_merge_seq, smallVectorAscending) {
  bool order = true;

  std::vector<int> in = koshkin_n_shell_sort_batchers_even_odd_merge_seq::GetRandomVector(15);
  std::vector<int> out(in.size(), 0);

  std::vector<int> res = in;
  std::ranges::sort(res.begin(), res.end());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&order));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, res);
}

TEST(koshkin_n_shell_sort_batchers_even_odd_merge_seq, bigVectorDescending) {
  bool order = false;

  std::vector<int> in = koshkin_n_shell_sort_batchers_even_odd_merge_seq::GetRandomVector(1500);
  std::vector<int> out(in.size(), 0);

  std::vector<int> res = in;
  std::ranges::sort(res.begin(), res.end(), std::greater<>());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&order));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, res);
}

TEST(koshkin_n_shell_sort_batchers_even_odd_merge_seq, bigVectorAscending) {
  bool order = true;

  std::vector<int> in = koshkin_n_shell_sort_batchers_even_odd_merge_seq::GetRandomVector(1500);
  std::vector<int> out(in.size(), 0);

  std::vector<int> res = in;
  std::ranges::sort(res.begin(), res.end());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&order));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, res);
}

TEST(koshkin_n_shell_sort_batchers_even_odd_merge_seq, PrimeSizeVectorAscending) {
  bool order = true;

  std::vector<int> in = koshkin_n_shell_sort_batchers_even_odd_merge_seq::GetRandomVector(101);
  std::vector<int> out(in.size(), 0);

  std::vector<int> res = in;
  std::ranges::sort(res.begin(), res.end());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&order));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, res);
}

TEST(koshkin_n_shell_sort_batchers_even_odd_merge_seq, PrimeSizeVectorDescending) {
  bool order = false;

  std::vector<int> in = koshkin_n_shell_sort_batchers_even_odd_merge_seq::GetRandomVector(107);
  std::vector<int> out(in.size(), 0);

  std::vector<int> res = in;
  std::ranges::sort(res.begin(), res.end(), std::greater<>());

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&order));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  koshkin_n_shell_sort_batchers_even_odd_merge_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(out, res);
}