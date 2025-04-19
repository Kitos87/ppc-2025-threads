#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <tuple>
#include <vector>
#ifndef _WIN32
#include <opencv2/opencv.hpp>
#endif
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/voroshilov_v_convex_hull_components/include/chc.hpp"
#include "seq/voroshilov_v_convex_hull_components/include/chc_seq.hpp"

using namespace voroshilov_v_convex_hull_components_seq;

namespace {

std::vector<int> GenBinVec(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> bin_vec(size);

  for (int i = 0; i < size; i++) {
    bin_vec[i] = static_cast<int>(gen() % 2);
  }

  return bin_vec;
}

#ifndef _WIN32

std::vector<Hull> GetHullsWithOpencv(int height, int width, std::vector<int>& pixels) {
  cv::Mat binary_mat(height, width, CV_8U);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (pixels[(i * width) + j] == 0) {
        binary_mat.at<uchar>(i, j) = 0;
      } else {
        binary_mat.at<uchar>(i, j) = 255;
      }
    }
  }

  cv::Mat labels(height, width, CV_32S);
  int num_labels = cv::connectedComponents(binary_mat, labels, 8, CV_32S);

  std::vector<std::vector<cv::Point>> components_cv(num_labels - 1);
  for (int y = 0; y < height; ++y) {
    const int* row = reinterpret_cast<const int*>(labels.ptr(y));
    for (int x = 0; x < width; ++x) {
      int label = row[x];
      if (label > 0) {  // 0 is background
        components_cv[label - 1].emplace_back(x, y);
      }
    }
  }

  std::vector<Hull> hulls_cv;
  for (const auto& component_cv : components_cv) {
    if (component_cv.empty()) {
      continue;
    }
    std::vector<cv::Point> hull_cv;
    cv::convexHull(component_cv, hull_cv);

    Hull hull;
    for (const cv::Point& p : hull_cv) {
      Pixel pixel(p.y, p.x);
      hull.pixels.push_back(pixel);
    }
    hulls_cv.push_back(hull);
  }

  return hulls_cv;
}

#endif

void SortPixels(Hull& hull) {
  std::ranges::sort(hull.pixels,
                    [](const Pixel& p1, const Pixel& p2) { return std::tie(p1.y, p1.x) < std::tie(p2.y, p2.x); });
}

void SortHulls(std::vector<Hull>& hulls) {
  std::ranges::sort(hulls, [](const Hull& a, const Hull& b) {
    const Pixel& left_top_a = *std::ranges::min_element(
        a.pixels, [](const Pixel& p1, const Pixel& p2) { return p1.x < p2.x || (p1.x == p2.x && p1.y < p2.y); });
    const Pixel& left_top_b = *std::ranges::min_element(
        b.pixels, [](const Pixel& p1, const Pixel& p2) { return p1.x < p2.x || (p1.x == p2.x && p1.y < p2.y); });

    return left_top_a.x < left_top_b.x || (left_top_a.x == left_top_b.x && left_top_a.y < left_top_b.y);
  });
}

bool IsHullSubset(Hull& hull_first, Hull& hull_second) {
  Hull smaller;
  Hull larger;
  if (hull_first.pixels.size() <= hull_second.pixels.size()) {
    smaller = hull_first;
    larger = hull_second;
  } else {
    smaller = hull_second;
    larger = hull_first;
  }

  SortPixels(smaller);
  SortPixels(larger);

  size_t i = 0;
  size_t j = 0;

  while (i < smaller.pixels.size() && j < larger.pixels.size()) {
    if (smaller.pixels[i] == larger.pixels[j]) {
      i++;  // found a point from smaller in larger
    }
    j++;  // move on larger
  }

  return i == smaller.pixels.size();  // if true then smaller is subset of larger
}

}  // namespace

TEST(voroshilov_v_convex_hull_components_seq, chc_pipeline_run) {
  std::vector<int> pixels = GenBinVec(10'000'000);
  int height = 10'000;
  int width = 1'000;

  int* p_height = &height;
  int* p_width = &width;
  std::vector<int> hulls_indexes_out(height * width);
  std::vector<int> pixels_indexes_out(height * width);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(p_height));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(p_width));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(pixels.data()));
  task_data_seq->inputs_count.emplace_back(pixels.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(hulls_indexes_out.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(pixels_indexes_out.data()));
  task_data_seq->outputs_count.emplace_back(0);

  auto chc_task_sequential =
      std::make_shared<voroshilov_v_convex_hull_components_seq::ChcTaskSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(chc_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  int hulls_size = static_cast<int>(task_data_seq->outputs_count[0]);
  std::vector<Hull> hulls = UnpackHulls(hulls_indexes_out, pixels_indexes_out, height, width, hulls_size);

#ifndef _WIN32
  std::vector<Hull> hulls_cv = GetHullsWithOpencv(height, width, pixels);

  SortHulls(hulls);
  for (Hull& hull : hulls) {
    SortPixels(hull);
  }

  SortHulls(hulls_cv);
  for (Hull& hull_cv : hulls_cv) {
    SortPixels(hull_cv);
  }

  ASSERT_EQ(hulls.size(), hulls_cv.size());

  for (size_t i = 0; i < hulls.size(); i++) {
    EXPECT_TRUE(IsHullSubset(hulls[i], hulls_cv[i]));
  }

#endif
}

TEST(voroshilov_v_convex_hull_components_seq, chc_task_run) {
  std::vector<int> pixels = GenBinVec(10'000'000);
  int height = 10'000;
  int width = 1'000;

  int* p_height = &height;
  int* p_width = &width;
  std::vector<int> hulls_indexes_out(height * width);
  std::vector<int> pixels_indexes_out(height * width);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(p_height));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(p_width));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(pixels.data()));
  task_data_seq->inputs_count.emplace_back(pixels.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(hulls_indexes_out.data()));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(pixels_indexes_out.data()));
  task_data_seq->outputs_count.emplace_back(0);

  auto chc_task_sequential =
      std::make_shared<voroshilov_v_convex_hull_components_seq::ChcTaskSequential>(task_data_seq);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(chc_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  int hulls_size = static_cast<int>(task_data_seq->outputs_count[0]);
  std::vector<Hull> hulls = UnpackHulls(hulls_indexes_out, pixels_indexes_out, height, width, hulls_size);

#ifndef _WIN32
  std::vector<Hull> hulls_cv = GetHullsWithOpencv(height, width, pixels);

  SortHulls(hulls);
  for (Hull& hull : hulls) {
    SortPixels(hull);
  }

  SortHulls(hulls_cv);
  for (Hull& hull_cv : hulls_cv) {
    SortPixels(hull_cv);
  }

  ASSERT_EQ(hulls.size(), hulls_cv.size());

  for (size_t i = 0; i < hulls.size(); i++) {
    EXPECT_TRUE(IsHullSubset(hulls[i], hulls_cv[i]));
  }
#endif
}
