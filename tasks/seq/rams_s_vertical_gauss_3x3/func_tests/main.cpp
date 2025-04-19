#include "seq/rams_s_vertical_gauss_3x3/include/main.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#ifndef _WIN32
#include <opencv2/opencv.hpp>
#endif
#include <tuple>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"

class RamsSVerticalGauss3x3SeqTest
    : public ::testing::TestWithParam<
          std::tuple<uint32_t, uint32_t, std::vector<uint8_t>, std::vector<float>, std::vector<uint8_t>>> {};

namespace {
void RunTest(uint32_t width, uint32_t height, std::vector<uint8_t>& in, std::vector<float>& kernel,
             std::vector<uint8_t>& expected) {
  std::vector<uint8_t> out(expected.size());

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  rams_s_vertical_gauss_3x3_seq::TaskSequential test_task(task_data);
  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();
  EXPECT_EQ(out, expected);
}
}  // namespace

TEST_P(RamsSVerticalGauss3x3SeqTest, p) {
  auto [width, height, in, kernel, expected] = GetParam();
  RunTest(width, height, in, kernel, expected);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P( // NOLINT(misc-use-anonymous-namespace)
  rams_s_vertical_gauss_3x3_seq_test,
  RamsSVerticalGauss3x3SeqTest,
  ::testing::Values(
    std::tuple(
      0, 0,
      std::vector<uint8_t>{},
      std::vector<float>{
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
      },
      std::vector<uint8_t>{}
    ),

    std::tuple(
      1, 1,
      std::vector<uint8_t>{255,255,255},
      std::vector<float>{
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
      },
      std::vector<uint8_t>{255,255,255}
    ),

    std::tuple(
      1, 3,
      std::vector<uint8_t>{
        255,255,255,
        200,200,200,
        100,100,100,
      },
      std::vector<float>{
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
      },
      std::vector<uint8_t>{
        255,255,255,
        200,200,200,
        100,100,100,
      }
    ),

    std::tuple(
      2, 2,
      std::vector<uint8_t>{
        255, 0, 0, 255, 0, 0,
        255, 0, 0, 255, 0, 0
      },
      std::vector<float>{
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
      },
      std::vector<uint8_t>{
        255, 0, 0, 255, 0, 0,
        255, 0, 0, 255, 0, 0
      }
    ),

    std::tuple(
      4, 4,
      std::vector<uint8_t>{
          1, 200, 188,       2,  50,  67,       3,  99,  21,       4, 255,   0,
          5,  31,  97,     199,  86,  37,      27,  88, 191,     238,  99, 114,
          0,  89,  66,      98,  43,  51,     201,  67, 149,     134,  55,  86,
        234,   6,  16,     166,  86,  78,     144,  97, 206,     109,  78, 197
      },
      std::vector<float>{
        0, 0, 0,
        0, 1, 0,
        0, 0, 0
      },
      std::vector<uint8_t>{
          1, 200, 188,       2,  50,  67,       3,  99,  21,       4, 255,   0,
          5,  31,  97,     199,  86,  37,      27,  88, 191,     238,  99, 114,
          0,  89,  66,      98,  43,  51,     201,  67, 149,     134,  55,  86,
        234,   6,  16,     166,  86,  78,     144,  97, 206,     109,  78, 197
      }
    ),

    std::tuple(
      4, 4,
      std::vector<uint8_t>{
          1, 200, 188,       2,  50,  67,       3,  99,  21,       4, 255,   0,
          5,  31,  97,     199,  86,  37,      27,  88, 191,     238,  99, 114,
          0,  89,  66,      98,  43,  51,     201,  67, 149,     134,  55,  86,
        234,   6,  16,     166,  86,  78,     144,  97, 206,     109,  78, 197
      },
      std::vector<float>{
        0, 0, 1,
        0, 1, 0,
        0, 0, 1
      },
      std::vector<uint8_t>{
          1, 200, 188,       2,  50,  67,       3,  99,  21,       4, 255,   0,
          5,  31,  97,     255, 252, 207,     165, 255, 255,     238,  99, 114,
          0,  89,  66,     255, 228, 255,     255, 244, 255,     134,  55,  86,
        234,   6,  16,     166,  86,  78,     144,  97, 206,     109,  78, 197
      }
    ),

    std::tuple(
      4, 4,
      std::vector<uint8_t>{
          1, 200, 188,       2,  50,  67,       3,  99,  21,       4, 255,   0,
          5,  31,  97,     199,  86,  37,      27,  88, 191,     238,  99, 114,
          0,  89,  66,      98,  43,  51,     201,  67, 149,     134,  55,  86,
        234,   6,  16,     166,  86,  78,     144,  97, 206,     109,  78, 197
      },
      std::vector<float>{
        1.0/16, 1.0/8, 1.0/16,
        1.0/8,  1.0/4, 1.0/8,
        1.0/16, 1.0/8, 1.0/16
      },
      std::vector<uint8_t>{
         1, 200, 188,        2,  50,  67,       3,  99,  21,       4, 255,   0,
         5,  31,  97,       79,  76,  87,     102,  91, 101,     238,  99, 114,
         0,  89,  66,      121,  66,  86,     145,  74, 131,     134,  55,  86,
       234,   6,  16,      166,  86,  78,     144,  97, 206,     109,  78, 197,
      }
    )
  )
);
// clang-format on

#ifndef _WIN32
TEST(rams_s_vertical_gauss_3x3_seq, test_with_fixture) {
  cv::Mat img = cv::imread(ppc::util::GetAbsolutePath("seq/rams_s_vertical_gauss_3x3/data/flower.png"));
  cv::Mat img_expected =
      cv::imread(ppc::util::GetAbsolutePath("seq/rams_s_vertical_gauss_3x3/data/flower-blurred.png"));
  std::vector<uint8_t> in(img.reshape(1, static_cast<int>(img.total()) * img.channels()));
  std::vector<uint8_t> expected(
      img_expected.reshape(1, static_cast<int>(img_expected.total()) * img_expected.channels()));
  // clang-format off
  std::vector<float> kernel{
    1.0/16, 1.0/8, 1.0/16,
    1.0/8,  1.0/4, 1.0/8,
    1.0/16, 1.0/8, 1.0/16
  };
  // clang-format on

  RunTest(img.cols, img.rows, in, kernel, expected);
}
#endif
