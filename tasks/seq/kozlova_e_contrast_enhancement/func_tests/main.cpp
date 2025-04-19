#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/kozlova_e_contrast_enhancement/include/ops_seq.hpp"

namespace {
std::vector<int> GenerateVector(int length);

std::vector<int> GenerateVector(int length) {
  std::vector<int> vec;
  vec.reserve(length);
  for (int i = 0; i < length; ++i) {
    vec.push_back(rand() % 256);
  }
  return vec;
}
}  // namespace

TEST(kozlova_e_contrast_enhancement_seq, test_1st_image) {
  std::vector<int> in{10, 0, 50, 100, 200, 34};
  size_t width = 2;
  size_t height = 3;
  std::vector<int> out(6, 0);
  std::vector<int> expected{12, 0, 63, 127, 255, 43};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs_count.emplace_back(width);
  task_data_seq->inputs_count.emplace_back(height);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kozlova_e_contrast_enhancement_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_EQ(out[i], expected[i]);
  }
}

TEST(kozlova_e_contrast_enhancement_seq, test_image2) {
  int size = 400;
  std::vector<int> in = GenerateVector(size);
  size_t width = 10;
  size_t height = 40;
  std::vector<int> out(size, 0);
  std::vector<int> expect(size, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs_count.emplace_back(width);
  task_data_seq->inputs_count.emplace_back(height);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kozlova_e_contrast_enhancement_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  int min_value = *std::ranges::min_element(in);
  int max_value = *std::ranges::max_element(in);

  for (size_t i = 0; i < in.size(); ++i) {
    int expected = static_cast<int>(((in[i] - min_value) / (double)(max_value - min_value)) * 255);
    expected = std::clamp(expected, 0, 255);
    EXPECT_EQ(out[i], expected);
  }
}

TEST(kozlova_e_contrast_enhancement_seq, test_empty_input) {
  std::vector<int> in = {};
  size_t width = 0;
  size_t height = 0;
  std::vector<int> out(0, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs_count.emplace_back(width);
  task_data_seq->inputs_count.emplace_back(height);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kozlova_e_contrast_enhancement_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_FALSE(test_task_sequential.Validation());
}

TEST(kozlova_e_contrast_enhancement_seq, test_same_values_input) {
  std::vector<int> in(6, 100);
  size_t width = 2;
  size_t height = 3;
  std::vector<int> out(6, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs_count.emplace_back(width);
  task_data_seq->inputs_count.emplace_back(height);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kozlova_e_contrast_enhancement_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_EQ(in, out);
}

TEST(kozlova_e_contrast_enhancement_seq, test_difference_input) {
  std::vector<int> in{10, 20, 30, 100, 200, 250};
  size_t width = 2;
  size_t height = 3;
  std::vector<int> out(6, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs_count.emplace_back(width);
  task_data_seq->inputs_count.emplace_back(height);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kozlova_e_contrast_enhancement_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  int min_value = *std::ranges::min_element(in);
  int max_value = *std::ranges::max_element(in);

  for (size_t i = 0; i < in.size(); ++i) {
    int expected = static_cast<int>(((in[i] - min_value) / (double)(max_value - min_value)) * 255);
    expected = std::clamp(expected, 0, 255);
    EXPECT_EQ(out[i], expected);
  }
}

TEST(kozlova_e_contrast_enhancement_seq, test_negative_values) {
  std::vector<int> in{-10, -20, -30, -100, -200, -250};
  size_t width = 3;
  size_t height = 2;
  std::vector<int> out(6, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs_count.emplace_back(width);
  task_data_seq->inputs_count.emplace_back(height);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kozlova_e_contrast_enhancement_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  ASSERT_ANY_THROW(test_task_sequential.Run());
}

TEST(kozlova_e_contrast_enhancement_seq, test_incorrect_input_size) {
  std::vector<int> in = {3, 3, 3};
  size_t width = 3;
  size_t height = 1;
  std::vector<int> out(0, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs_count.emplace_back(width);
  task_data_seq->inputs_count.emplace_back(height);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kozlova_e_contrast_enhancement_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_FALSE(test_task_sequential.Validation());
}

TEST(kozlova_e_contrast_enhancement_seq, test_incorrect_input_width) {
  std::vector<int> in = {3, 3, 3, 3};
  size_t width = 3;
  size_t height = 1;
  std::vector<int> out(4, 0);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs_count.emplace_back(width);
  task_data_seq->inputs_count.emplace_back(height);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  kozlova_e_contrast_enhancement_seq::TestTaskSequential test_task_sequential(task_data_seq);
  ASSERT_FALSE(test_task_sequential.Validation());
}
