#pragma once

#include <cstddef>
#include <memory>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace krylov_m_monte_carlo {

using Point = std::span<double>;
using MathFunction = double (*)(const Point&);  // <-- avoid extra indirection [std::function<double(const Point&)>]
using Bound = std::pair<double, double>;

struct IntegrationParams {
  MathFunction func;
  std::vector<Bound> bounds;
  std::size_t iterations;

  [[nodiscard]] std::size_t Dimensions() const noexcept { return bounds.size(); }

  //

  std::shared_ptr<ppc::core::TaskData> CreateTaskData(double& result) {
    constexpr auto kUglyUnsafeTaskDataAddr = []<typename T>(T& o) {
      return reinterpret_cast<decltype(std::declval<ppc::core::TaskData>().inputs)::value_type>(&o);
    };

    auto task_data = std::make_shared<ppc::core::TaskData>();

    task_data->inputs = {kUglyUnsafeTaskDataAddr(*this)};
    task_data->inputs_count = {1, 1, 2, 2};
    //
    task_data->outputs = {kUglyUnsafeTaskDataAddr(result)};
    task_data->outputs_count = {1};

    return task_data;
  }

  static IntegrationParams& FromTaskData(ppc::core::TaskData& task_data) noexcept {
    return *reinterpret_cast<IntegrationParams*>(task_data.inputs[0]);
  }
  static double& OutputOf(ppc::core::TaskData& task_data) noexcept {
    return *reinterpret_cast<double*>(task_data.outputs[0]);
  }

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version) {  // NOLINT(readability-identifier-naming)
    ar & bounds;
    ar & iterations;
  }
};

class TaskCommon : public ppc::core::Task {
 public:
  explicit TaskCommon(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override = 0;
  bool PostProcessingImpl() override;

 protected:
  void ApplyParams();

  IntegrationParams* params;
  double res;

  double vol;
  std::vector<std::uniform_real_distribution<double>> dists;
};

}  // namespace krylov_m_monte_carlo