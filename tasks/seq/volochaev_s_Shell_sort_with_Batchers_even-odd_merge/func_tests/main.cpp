#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/volochaev_s_Shell_sort_with_Batchers_even-odd_merge/include/ops_seq.hpp"

namespace {
void GetRandomVector(std::vector<int> &v, int a, int b) {
  std::random_device dev;
  std::mt19937 gen(dev());

  if (a >= b) {
    throw std::invalid_argument("error.");
  }

  std::uniform_int_distribution<> dis(a, b);

  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = dis(gen);
  }
}
}  // namespace

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_error_in_val) {
  constexpr size_t kSizeOfVector = 0;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  std::vector<int> out(kSizeOfVector, 0);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), false);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_error_in_generate) {
  constexpr size_t kSizeOfVector = 100;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  ASSERT_ANY_THROW(GetRandomVector(in, 1000, -1000));
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_small_vector) {
  constexpr size_t kSizeOfVector = 100;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_small_vector2) {
  constexpr size_t kSizeOfVector = 200;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_small_vector3) {
  constexpr size_t kSizeOfVector = 300;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_small_vector4) {
  constexpr size_t kSizeOfVector = 400;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_medium_vector) {
  constexpr size_t kSizeOfVector = 500;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_medium_vector2) {
  constexpr size_t kSizeOfVector = 600;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_medium_vector3) {
  constexpr size_t kSizeOfVector = 700;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_medium_vector4) {
  constexpr size_t kSizeOfVector = 800;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_medium_vector5) {
  constexpr size_t kSizeOfVector = 900;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_big_vector) {
  constexpr size_t kSizeOfVector = 1000;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_big_vector2) {
  constexpr size_t kSizeOfVector = 2000;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_big_vector3) {
  constexpr size_t kSizeOfVector = 3000;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_big_vector4) {
  constexpr size_t kSizeOfVector = 4000;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_extra_big_vector) {
  constexpr size_t kSizeOfVector = 10000;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_prime_size_vector) {
  constexpr size_t kSizeOfVector = 7;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_prime_size_vector1) {
  constexpr size_t kSizeOfVector = 13;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_prime_size_vector2) {
  constexpr size_t kSizeOfVector = 17;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_prime_size_vector3) {
  constexpr size_t kSizeOfVector = 23;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_prime_size_vector4) {
  constexpr size_t kSizeOfVector = 29;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_odd_number_of_elements) {
  constexpr size_t kSizeOfVector = 101;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_odd_number_of_elements1) {
  constexpr size_t kSizeOfVector = 99;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_odd_number_of_elements2) {
  constexpr size_t kSizeOfVector = 201;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_odd_number_of_elements3) {
  constexpr size_t kSizeOfVector = 199;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_odd_number_of_elements4) {
  constexpr size_t kSizeOfVector = 301;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_odd_number_of_elements5) {
  constexpr size_t kSizeOfVector = 299;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_odd_number_of_elements6) {
  constexpr size_t kSizeOfVector = 401;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_odd_number_of_elements7) {
  constexpr size_t kSizeOfVector = 399;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}

TEST(volochaev_s_Shell_sort_with_Batchers_even_odd_merge_seq, test_with_reverse) {
  constexpr size_t kSizeOfVector = 399;

  // Create data
  std::vector<int> in(kSizeOfVector, 0);
  GetRandomVector(in, -100, 100);
  std::vector<int> out(kSizeOfVector, 0);
  std::vector<int> answer(in);
  std::ranges::sort(answer);
  std::ranges::sort(in);
  std::ranges::reverse(in);

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  volochaev_s_shell_sort_with_batchers_even_odd_merge_seq::ShellSortSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();
  EXPECT_EQ(answer, out);
}