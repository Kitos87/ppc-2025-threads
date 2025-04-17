#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/petrov_a_radix_double_batcher/include/ops_seq.hpp"

namespace {
std::vector<double> RandomVector(size_t size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dist(-5000, 8000);
  std::vector<double> vec(size);
  std::ranges::generate(vec, [&dist, &gen] { return dist(gen); });
  return vec;
}

void STest(size_t size) {
  auto in = RandomVector(size);
  std::vector<double> out(in.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = petrov_a_radix_double_batcher_seq::TestTaskSequential(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_TRUE(std::ranges::is_sorted(out));
}

void STestR(size_t size) {
  std::vector<double> in(size);
  std::iota(in.rbegin(), in.rend(), 0);
  std::vector<double> out(in.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto task = petrov_a_radix_double_batcher_seq::TestTaskSequential(task_data);
  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  ASSERT_TRUE(std::ranges::is_sorted(out));
}
}  // namespace

TEST(petrov_a_radix_double_batcher_seq, test_0) { STest(0); }
TEST(petrov_a_radix_double_batcher_seq, test_1) { STest(1); }
TEST(petrov_a_radix_double_batcher_seq, test_2) { STest(2); }
TEST(petrov_a_radix_double_batcher_seq, test_3) { STest(3); }
TEST(petrov_a_radix_double_batcher_seq, test_4) { STest(4); }
TEST(petrov_a_radix_double_batcher_seq, test_5) { STest(5); }
TEST(petrov_a_radix_double_batcher_seq, test_6) { STest(6); }
TEST(petrov_a_radix_double_batcher_seq, test_7) { STest(7); }
TEST(petrov_a_radix_double_batcher_seq, test_8) { STest(8); }
TEST(petrov_a_radix_double_batcher_seq, test_9) { STest(9); }
TEST(petrov_a_radix_double_batcher_seq, test_10) { STest(10); }
TEST(petrov_a_radix_double_batcher_seq, test_11) { STest(11); }
TEST(petrov_a_radix_double_batcher_seq, test_111) { STest(111); }
TEST(petrov_a_radix_double_batcher_seq, test_213) { STest(213); }

TEST(petrov_a_radix_double_batcher_seq, test_inv_0) { STestR(0); }
TEST(petrov_a_radix_double_batcher_seq, test_inv_1) { STestR(1); }
TEST(petrov_a_radix_double_batcher_seq, test_inv_2) { STestR(2); }
TEST(petrov_a_radix_double_batcher_seq, test_inv_3) { STestR(3); }
TEST(petrov_a_radix_double_batcher_seq, test_inv_4) { STestR(4); }
TEST(petrov_a_radix_double_batcher_seq, test_inv_5) { STestR(5); }
TEST(petrov_a_radix_double_batcher_seq, test_inv_6) { STestR(6); }
TEST(petrov_a_radix_double_batcher_seq, test_inv_7) { STestR(7); }
TEST(petrov_a_radix_double_batcher_seq, test_inv_8) { STestR(8); }
TEST(petrov_a_radix_double_batcher_seq, test_inv_9) { STestR(9); }
TEST(petrov_a_radix_double_batcher_seq, test_inv_10) { STestR(10); }
TEST(petrov_a_radix_double_batcher_seq, test_inv_11) { STestR(11); }
TEST(petrov_a_radix_double_batcher_seq, test_inv_111) { STestR(111); }
TEST(petrov_a_radix_double_batcher_seq, test_inv_213) { STestR(213); }