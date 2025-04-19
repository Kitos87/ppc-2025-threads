#include "all/rams_s_vertical_gauss_3x3/include/main.hpp"

#include <algorithm>
#include <boost/serialization/vector.hpp>  // NOLINT(misc-include-cleaner)
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "boost/mpi/collectives/gatherv.hpp"
#include "boost/mpi/collectives/scatterv.hpp"
#include "core/util/include/util.hpp"

bool rams_s_vertical_gauss_3x3_all::TaskAll::PreProcessingImpl() {
  if (world_.rank() == 0) {
    width_ = task_data->inputs_count[0];
    height_ = task_data->inputs_count[1];
    input_ = std::vector<uint8_t>(task_data->inputs[0], task_data->inputs[0] + (height_ * width_ * 3));
    auto *k = reinterpret_cast<float *>(task_data->inputs[1]);
    kernel_ = std::vector<float>(k, k + task_data->inputs_count[2]);

    output_ = std::vector<uint8_t>(input_);
  }

  return true;
}

bool rams_s_vertical_gauss_3x3_all::TaskAll::ValidationImpl() {
  return world_.rank() != 0 ||
         (task_data->inputs_count[2] == 9 &&
          (task_data->inputs_count[0] * task_data->inputs_count[1] * 3) == task_data->outputs_count[0]);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool rams_s_vertical_gauss_3x3_all::TaskAll::RunImpl() {
  boost::mpi::broadcast(world_, height_, 0);
  boost::mpi::broadcast(world_, width_, 0);
  boost::mpi::broadcast(world_, kernel_, 0);

  if (height_ < 3 || width_ < 3) {
    return true;
  }

  std::size_t world_size = std::min(std::size_t(world_.size()), std::size_t(width_) - 2);
  if (std::size_t(world_.rank()) >= world_size) {
    world_.split(1);
    return true;
  }
  auto group = world_.split(0);
  std::size_t avg_to_send = width_ / world_size;
  std::size_t extra_to_send = width_ % world_size;
  std::vector<int> sendcounts(world_size);
  std::vector<int> recvcounts(world_size);
  std::vector<int> displs(world_size, 0);
  for (std::size_t i = 0; i < world_size; i++) {
    int padding = [&] {
      if (i != 0 && i != world_size - 1) {
        return 2;
      }
      return world_size == 1 ? 0 : 1;
    }();
    sendcounts[i] = static_cast<int>(avg_to_send + (i < extra_to_send ? 1 : 0) + padding) * 3;
    recvcounts[i] = static_cast<int>(avg_to_send + (i < extra_to_send ? 1 : 0) + padding - 2) * 3;
    if (i > 0) {
      displs[i] = displs[i - 1] + sendcounts[i - 1] - 2 * 3;
    }
  }

  int local_width = sendcounts[group.rank()] / 3;
  std::vector<uint8_t> local_input(local_width * height_ * 3);
  std::vector<uint8_t> local_output(local_width * height_ * 3);

  for (std::size_t y = 0; y < height_; y++) {
    boost::mpi::scatterv(group, input_.data() + (y * width_ * 3), sendcounts, displs,
                         local_input.data() + (y * local_width * 3), local_width * 3, 0);
  }

  /////

  const std::size_t num_threads = std::min(ppc::util::GetPPCNumThreads(), local_width);
  std::vector<std::thread> threads(num_threads);
  for (std::size_t thread_i = 0; thread_i < num_threads; thread_i++) {
    threads[thread_i] = std::thread([&, thread_i] {
      std::size_t amount = (local_width / num_threads) + (thread_i < local_width % num_threads ? 1 : 0);
      std::size_t left = ((local_width / num_threads) * thread_i) + std::min(local_width % num_threads, thread_i);
      std::size_t right = std::min(left + amount, std::size_t(local_width) - 1);
      for (std::size_t x = std::max(left, std::size_t(1)); x < right; x++) {
        for (std::size_t y = 1; y < height_ - 1; y++) {
          for (std::size_t i = 0; i < 3; i++) {
            local_output[((y * local_width + x) * 3) + i] = std::clamp(static_cast<int>(std::round(
#define INNER(Y_SHIFT, X_SHIFT) \
  local_input[((((y + (Y_SHIFT)) * local_width) + x + (X_SHIFT)) * 3) + i] * kernel_[4 + (3 * (Y_SHIFT)) + (X_SHIFT)]
#define OUTER(Y) (INNER(Y, -1) + INNER(Y, 0) + INNER(Y, 1))
                                                                           (OUTER(-1) + OUTER(0) + OUTER(1))
#undef OUTER
#undef INNER
                                                                               )),
                                                                       0, 255);
          }
        }
      }
    });
  }
  for (std::size_t thread_i = 0; thread_i < num_threads; thread_i++) {
    threads[thread_i].join();
  }

  /////

  int local_out_width = recvcounts[group.rank()];
  for (std::size_t y = 1; y < height_ - 1; y++) {
    boost::mpi::gatherv(group, local_output.data() + ((y * local_width + 1) * 3), local_out_width,
                        output_.data() + ((y * width_ + 1) * 3), recvcounts, 0);
  }

  return true;
}

bool rams_s_vertical_gauss_3x3_all::TaskAll::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::ranges::copy(output_, task_data->outputs[0]);
  }
  return true;
}
