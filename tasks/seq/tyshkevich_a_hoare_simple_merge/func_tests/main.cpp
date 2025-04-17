#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/tyshkevich_a_hoare_simple_merge/include/ops_seq.hpp"

namespace {
template <typename T>
std::vector<T> GenRandVec(size_t size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dist(-1000, 1000);

  std::vector<T> vec(size);
  std::ranges::generate(vec, [&] { return dist(gen); });

  return vec;
}

template <typename T, typename Comparator>
void TestSort(std::vector<T> &&in, Comparator cmp) {
  std::vector<T> out(in.size());

  auto dat = std::make_shared<ppc::core::TaskData>();
  dat->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  dat->inputs_count.emplace_back(in.size());
  dat->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  dat->outputs_count.emplace_back(out.size());

  auto tt = tyshkevich_a_hoare_simple_merge_seq::CreateHoareTestTask<T>(dat, cmp);
  ASSERT_TRUE(tt.Validation());
  tt.PreProcessing();
  tt.Run();
  tt.PostProcessing();

  ASSERT_TRUE(std::ranges::is_sorted(out, cmp));
}

template <typename T, typename Comparator>
void TestSort(std::size_t size, Comparator cmp) {
  TestSort(GenRandVec<T>(size), cmp);
}
}  // namespace

TEST(tyshkevich_a_hoare_simple_merge_seq, invalid) {
  std::vector<int> in(5);
  std::vector<int> out(in.size() + 1);

  auto dat = std::make_shared<ppc::core::TaskData>();
  dat->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  dat->inputs_count.emplace_back(in.size());
  dat->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  dat->outputs_count.emplace_back(out.size());

  auto tt = tyshkevich_a_hoare_simple_merge_seq::CreateHoareTestTask<int>(dat, std::greater<>());
  ASSERT_FALSE(tt.Validation());
}

TEST(tyshkevich_a_hoare_simple_merge_seq, test_0_gt) { TestSort<int>(0, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_seq, test_0_lt) { TestSort<int>(0, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_seq, test_1_gt) { TestSort<int>(1, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_seq, test_1_lt) { TestSort<int>(1, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_seq, test_2_gt) { TestSort<int>(2, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_seq, test_2_lt) { TestSort<int>(2, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_seq, test_3_gt) { TestSort<int>(3, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_seq, test_3_lt) { TestSort<int>(3, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_seq, test_5_gt) { TestSort<int>(5, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_seq, test_5_lt) { TestSort<int>(5, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_seq, test_7_gt) { TestSort<int>(7, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_seq, test_7_lt) { TestSort<int>(7, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_seq, test_9_gt) { TestSort<int>(9, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_seq, test_9_lt) { TestSort<int>(9, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_seq, test_10_gt) { TestSort<int>(10, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_seq, test_10_lt) { TestSort<int>(10, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_seq, test_11_gt) { TestSort<int>(11, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_seq, test_11_lt) { TestSort<int>(11, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_seq, test_13_gt) { TestSort<int>(13, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_seq, test_13_lt) { TestSort<int>(13, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_seq, test_19_gt) { TestSort<int>(13, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_seq, test_19_lt) { TestSort<int>(13, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_seq, test_23_gt) { TestSort<int>(13, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_seq, test_23_lt) { TestSort<int>(13, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_seq, test_31_gt) { TestSort<int>(13, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_seq, test_31_lt) { TestSort<int>(13, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_seq, test_64_gt) { TestSort<int>(13, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_seq, test_64_lt) { TestSort<int>(13, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_seq, test_100_gt) { TestSort<int>(13, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_seq, test_100_lt) { TestSort<int>(13, std::less<>()); }

TEST(tyshkevich_a_hoare_simple_merge_seq, test_homogeneous_gt) { TestSort<int>({1, 1, 1}, std::greater<>()); }
TEST(tyshkevich_a_hoare_simple_merge_seq, test_homogeneous_lt) { TestSort<int>({1, 1, 1}, std::less<>()); }
