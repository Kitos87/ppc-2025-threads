#include "all/rams_s_vertical_gauss_3x3/include/main.hpp"

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#ifndef _WIN32
#include <opencv2/opencv.hpp>
#endif
#include <random>
#include <tuple>
#include <vector>

#include "all/rams_s_vertical_gauss_3x3/include/main_seq.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"

class RamsSVerticalGauss3x3AllTest
    : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t, std::vector<uint8_t>, std::vector<float>>> {};

namespace {
void RunTest(uint32_t width, uint32_t height, std::vector<uint8_t>& in, std::vector<float>& kernel) {
  boost::mpi::communicator world;
  std::vector<uint8_t> out(in.size());

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(width);
  task_data->inputs_count.emplace_back(height);
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data->inputs_count.emplace_back(kernel.size());
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(out.size());

  rams_s_vertical_gauss_3x3_all::TaskAll test_task(task_data);
  ASSERT_EQ(test_task.Validation(), true);
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  if (world.rank() == 0) {
    std::vector<uint8_t> expected(in.size());

    {
      std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();
      task_data_seq->inputs.emplace_back(in.data());
      task_data_seq->inputs_count.emplace_back(width);
      task_data_seq->inputs_count.emplace_back(height);
      task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
      task_data_seq->inputs_count.emplace_back(kernel.size());
      task_data_seq->outputs.emplace_back(expected.data());
      task_data_seq->outputs_count.emplace_back(expected.size());

      rams_s_vertical_gauss_3x3_seq::TaskSequential test_task_seq(task_data_seq);
      ASSERT_EQ(test_task_seq.Validation(), true);
      test_task_seq.PreProcessing();
      test_task_seq.Run();
      test_task_seq.PostProcessing();
    }

    EXPECT_EQ(out, expected);
  }
}

std::vector<uint8_t> GenerateRandomImage(std::size_t width, std::size_t height) {
  std::vector<uint8_t> in(width * height * 3);
  std::random_device dev;
  std::mt19937 gen(dev());
  for (std::size_t x = 0; x < width; x++) {
    for (std::size_t y = 0; y < height; y++) {
      for (std::size_t i = 0; i < 3; i++) {
        std::size_t k = 0;
        while ((k = gen()) == 0) {
        }
        in[((y * width + x) * 3) + i] = k % 256;
      }
    }
  }
  return in;
}
}  // namespace

TEST_P(RamsSVerticalGauss3x3AllTest, p) {
  auto [width, height, in, kernel] = GetParam();
  RunTest(width, height, in, kernel);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P( // NOLINT(misc-use-anonymous-namespace)
  rams_s_vertical_gauss_3x3_all_test,
  RamsSVerticalGauss3x3AllTest,
  ::testing::Values(
    std::tuple(
      0, 0,
      std::vector<uint8_t>{},
      std::vector<float>{
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
      }
    ),

    std::tuple(
      1, 1,
      std::vector<uint8_t>{255,255,255},
      std::vector<float>{
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
      }
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
      }
    ),

    std::tuple(
      3, 4,
      GenerateRandomImage(3, 4),
      std::vector<float>{
        1.0/16, 1.0/8, 1.0/16,
        1.0/8,  1.0/4, 1.0/8,
        1.0/16, 1.0/8, 1.0/16
      }
    ),

    std::tuple(
      7, 7,
      GenerateRandomImage(7, 7),
      std::vector<float>{
        1.0/16, 1.0/8, 1.0/16,
        1.0/8,  1.0/4, 1.0/8,
        1.0/16, 1.0/8, 1.0/16
      }
    ),

    std::tuple(
      9, 3,
      GenerateRandomImage(9, 3),
      std::vector<float>{
        1.0/16, 1.0/8, 1.0/16,
        1.0/8,  1.0/4, 1.0/8,
        1.0/16, 1.0/8, 1.0/16
      }
    ),

    std::tuple(
      15, 11,
      GenerateRandomImage(15, 11),
      std::vector<float>{
        1.0/16, 1.0/8, 1.0/16,
        1.0/8,  1.0/4, 1.0/8,
        1.0/16, 1.0/8, 1.0/16
      }
    ),

    std::tuple(
      58, 51,
      GenerateRandomImage(58, 51),
      std::vector<float>{
        1.0/16, 1.0/8, 1.0/16,
        1.0/8,  1.0/4, 1.0/8,
        1.0/16, 1.0/8, 1.0/16
      }
    )
  )
);
// clang-format on

#ifndef _WIN32
TEST(rams_s_vertical_gauss_3x3_all, test_with_fixture) {
  cv::Mat img = cv::imread(ppc::util::GetAbsolutePath("all/rams_s_vertical_gauss_3x3/data/flower.png"));
  std::vector<uint8_t> in(img.reshape(1, static_cast<int>(img.total()) * img.channels()));
  // clang-format off
  std::vector<float> kernel{
    1.0/16, 1.0/8, 1.0/16,
    1.0/8,  1.0/4, 1.0/8,
    1.0/16, 1.0/8, 1.0/16
  };
  // clang-format on

  RunTest(img.cols, img.rows, in, kernel);
}
#endif
