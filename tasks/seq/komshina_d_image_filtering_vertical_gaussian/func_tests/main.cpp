#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/komshina_d_image_filtering_vertical_gaussian/include/ops_seq.hpp"

TEST(komshina_d_image_filtering_vertical_gaussian_seq, SinglePixelImage) {
  std::size_t width = 1;
  std::size_t height = 1;
  std::vector<unsigned char> in = {255, 255, 255};
  std::vector<float> kernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  std::vector<unsigned char> expected = {255, 255, 255};
  std::vector<unsigned char> out(expected.size());

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  komshina_d_image_filtering_vertical_gaussian_seq::TestTaskSequential test_task(task_data);
  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();
  EXPECT_EQ(out, expected);
}

TEST(komshina_d_image_filtering_vertical_gaussian_seq, EmptyImage) {
  std::size_t width = 0;
  std::size_t height = 0;
  std::vector<unsigned char> in = {};
  std::vector<float> kernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  std::vector<unsigned char> expected = {};
  std::vector<unsigned char> out(expected.size());

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  komshina_d_image_filtering_vertical_gaussian_seq::TestTaskSequential test_task(task_data);
  ASSERT_EQ(test_task.Validation(), false);
}

TEST(komshina_d_image_filtering_vertical_gaussian_seq, Small2x2Image) {
  std::size_t width = 2;
  std::size_t height = 2;
  std::vector<unsigned char> in = {100, 150, 200, 50, 100, 150, 200, 250, 100, 150, 200, 50};
  std::vector<float> kernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  std::vector<unsigned char> expected = in;
  std::vector<unsigned char> out(expected.size());

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  komshina_d_image_filtering_vertical_gaussian_seq::TestTaskSequential test_task(task_data);
  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();
  EXPECT_EQ(out, expected);
}

TEST(komshina_d_image_filtering_vertical_gaussian_seq, SingleRowImage) {
  std::size_t width = 3;
  std::size_t height = 1;
  std::vector<unsigned char> in = {10, 20, 30, 40, 50, 60, 70, 80, 90};
  std::vector<float> kernel = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<unsigned char> expected = in;
  std::vector<unsigned char> out(expected.size());

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  komshina_d_image_filtering_vertical_gaussian_seq::TestTaskSequential test_task(task_data);
  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();
  EXPECT_EQ(out, expected);
}

TEST(komshina_d_image_filtering_vertical_gaussian_seq, ZeroWidthImage) {
  std::size_t width = 0;
  std::size_t height = 3;
  std::vector<unsigned char> in = {255, 255, 255};
  std::vector<float> kernel = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<unsigned char> expected = {};
  std::vector<unsigned char> out(expected.size());

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  komshina_d_image_filtering_vertical_gaussian_seq::TestTaskSequential test_task(task_data);
  ASSERT_EQ(test_task.Validation(), false);
}

TEST(komshina_d_image_filtering_vertical_gaussian_seq, ValidationInvalidKernelSize) {
  std::size_t width = 3;
  std::size_t height = 3;
  std::vector<unsigned char> in = {255, 255, 255};
  std::vector<float> kernel = {1, 1};
  std::vector<unsigned char> out(9);

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  komshina_d_image_filtering_vertical_gaussian_seq::TestTaskSequential test_task(task_data);
  ASSERT_EQ(test_task.Validation(), false);
}

TEST(komshina_d_image_filtering_vertical_gaussian_seq, ValidationInvalidOutputSize) {
  std::size_t width = 3;
  std::size_t height = 3;
  std::vector<unsigned char> in = {255, 255, 255};
  std::vector<float> kernel = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<unsigned char> out(5);

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  komshina_d_image_filtering_vertical_gaussian_seq::TestTaskSequential test_task(task_data);
  ASSERT_EQ(test_task.Validation(), false);
}

TEST(komshina_d_image_filtering_vertical_gaussian_seq, PostProcessingCorrectness) {
  std::size_t width = 2;
  std::size_t height = 2;
  std::vector<unsigned char> in = {100, 150, 200, 50, 100, 150, 200, 250, 100, 150, 200, 50};
  std::vector<float> kernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  std::vector<unsigned char> expected = in;
  std::vector<unsigned char> out(expected.size());

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  komshina_d_image_filtering_vertical_gaussian_seq::TestTaskSequential test_task(task_data);
  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(out[i], expected[i]);
  }
}

TEST(komshina_d_image_filtering_vertical_gaussian_seq, RandomImage) {
  std::size_t width = 5;
  std::size_t height = 5;
  std::vector<unsigned char> in(width * height * 3);
  std::vector<float> kernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  std::vector<unsigned char> out(in.size());

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);

  for (size_t i = 0; i < in.size(); ++i) {
    in[i] = dis(gen);
  }

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  komshina_d_image_filtering_vertical_gaussian_seq::TestTaskSequential test_task(task_data);

  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_GE(out[i], 0);
    EXPECT_LE(out[i], 255);
  }
}

TEST(komshina_d_image_filtering_vertical_gaussian_seq, RandomImage2) {
  std::size_t width = 17;
  std::size_t height = 23;
  std::vector<unsigned char> in(width * height * 3);
  std::vector<float> kernel = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  std::vector<unsigned char> out(in.size());

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);

  for (size_t i = 0; i < in.size(); ++i) {
    in[i] = dis(gen);
  }

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  komshina_d_image_filtering_vertical_gaussian_seq::TestTaskSequential test_task(task_data);

  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_GE(out[i], 0);
    EXPECT_LE(out[i], 255);
  }
}