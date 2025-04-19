#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/milovankin_m_histogram_stretching/include/ops_seq.hpp"

namespace {

milovankin_m_histogram_stretching_seq::TestTaskSequential CreateTask(std::vector<uint8_t>& data_in,
                                                                     std::vector<uint8_t>& data_out) {
  auto task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(data_in.data());
  task_data->inputs_count.emplace_back(static_cast<uint32_t>(data_in.size()));

  task_data->outputs.emplace_back(data_out.data());
  task_data->outputs_count.emplace_back(static_cast<uint32_t>(data_out.size()));

  return milovankin_m_histogram_stretching_seq::TestTaskSequential(task_data);
}

}  // namespace

TEST(milovankin_m_histogram_stretching_seq, test_small_data) {
  // clang-format off
  std::vector<uint8_t> data_in = {
	50, 100, 100, 200,
	100, 50, 250, 100,
	100, 100, 200, 50,
	100, 200, 50, 250,
  };
  std::vector<uint8_t> data_expected = {
	0,   64,  64,  191,
    64,  0,   255, 64,
    64,  64,  191, 0,
    64,  191, 0,   255
  };
  // clang-format on
  std::vector<uint8_t> data_out(data_in.size());

  milovankin_m_histogram_stretching_seq::TestTaskSequential task = CreateTask(data_in, data_out);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  ASSERT_EQ(data_out, data_expected);
}

TEST(milovankin_m_histogram_stretching_seq, test_single_element) {
  std::vector<uint8_t> data_in = {150};
  std::vector<uint8_t> data_out(1);

  auto task = CreateTask(data_in, data_out);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_EQ(data_out[0], 150);
}

TEST(milovankin_m_histogram_stretching_seq, test_empty_data) {
  std::vector<uint8_t> data_in;
  std::vector<uint8_t> data_out;

  auto task = CreateTask(data_in, data_out);
  ASSERT_FALSE(task.Validation());
}

TEST(milovankin_m_histogram_stretching_seq, test_validation_fail_different_buffer_sizes) {
  std::vector<uint8_t> data_in(10, 100);
  std::vector<uint8_t> data_out(5);

  auto task = CreateTask(data_in, data_out);
  ASSERT_FALSE(task.Validation());
}

TEST(milovankin_m_histogram_stretching_seq, test_validation_fail_output_buffer_empty) {
  std::vector<uint8_t> data_in(10);
  std::vector<uint8_t> data_out;

  auto task = CreateTask(data_in, data_out);
  ASSERT_FALSE(task.Validation());
}

TEST(milovankin_m_histogram_stretching_seq, test_filled_image) {
  // clang-format off
  std::vector<uint8_t> data_in(100, 123);
  std::vector<uint8_t> data_expected(100, 123);
  // clang-format on
  std::vector<uint8_t> data_out(data_in.size());

  milovankin_m_histogram_stretching_seq::TestTaskSequential task = CreateTask(data_in, data_out);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  task.Run();
  task.PostProcessing();

  ASSERT_EQ(data_expected, data_out);
}

TEST(milovankin_m_histogram_stretching_seq, test_big_image) {
  std::vector<uint8_t> data_in(1024, 100);
  std::vector<uint8_t> data_out(data_in.size());

  data_in[0] = 50;
  data_in[1] = 125;

  std::vector<uint8_t> data_expected(data_in.size(), 170);
  data_expected[0] = 0;
  data_expected[1] = 255;

  milovankin_m_histogram_stretching_seq::TestTaskSequential task = CreateTask(data_in, data_out);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  task.Run();
  task.PostProcessing();

  ASSERT_EQ(data_expected, data_out);
}