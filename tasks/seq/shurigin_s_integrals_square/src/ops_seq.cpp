#include "seq/shurigin_s_integrals_square/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shurigin_s_integrals_square_seq {
Integral::Integral(std::shared_ptr<ppc::core::TaskData> task_data)
    : Task(task_data),
      down_limits_(1, 0.0),
      up_limits_(1, 0.0),
      counts_(1, 0),
      result_(0.0),
      func_(nullptr),
      dimensions_(1),
      task_data_(std::move(task_data)) {}

bool Integral::PreProcessingImpl() {
  try {
    if (!task_data_ || task_data_->inputs.empty() || task_data_->inputs[0] == nullptr) {
      throw std::invalid_argument("Invalid input data.");
    }
    auto* inputs = reinterpret_cast<double*>(task_data_->inputs[0]);
    if (dimensions_ > 1) {
      down_limits_.resize(dimensions_);
      up_limits_.resize(dimensions_);
      counts_.resize(dimensions_);

      for (int i = 0; i < dimensions_; ++i) {
        down_limits_[i] = inputs[i];
        up_limits_[i] = inputs[i + dimensions_];
        counts_[i] = static_cast<int>(inputs[i + (2 * dimensions_)]);

        if (counts_[i] <= 0) {
          throw std::invalid_argument("Number of intervals must be positive for all dimensions.");
        }
        if (up_limits_[i] <= down_limits_[i]) {
          throw std::invalid_argument("Upper limit must be greater than lower limit for all dimensions.");
        }
      }
    } else {
      down_limits_[0] = inputs[0];
      up_limits_[0] = inputs[1];
      counts_[0] = static_cast<int>(inputs[2]);

      if (counts_[0] <= 0) {
        throw std::invalid_argument("Number of intervals must be positive.");
      }
      if (up_limits_[0] <= down_limits_[0]) {
        throw std::invalid_argument("Upper limit must be greater than lower limit.");
      }
    }

    result_ = 0.0;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error in PreProcessingImpl: " << e.what() << '\n';
    return false;
  }
}

bool Integral::ValidationImpl() {
  try {
    if (!task_data_) {
      throw std::invalid_argument("task_data is null.");
    }
    if (task_data_->inputs_count.empty() || task_data_->outputs_count.empty()) {
      throw std::invalid_argument("Input or output counts are empty.");
    }

    size_t expected_input_size = 3 * dimensions_ * sizeof(double);
    if (task_data_->inputs_count[0] != expected_input_size) {
      throw std::invalid_argument("Expected " + std::to_string(3 * dimensions_) + " double values in input data.");
    }

    if (task_data_->outputs_count[0] != sizeof(double)) {
      throw std::invalid_argument("Expected one double value in output data.");
    }
    return true;
  } catch (const std::exception& e) {
    std::cerr << "error in ValidationImpl: " << e.what() << '\n';
    return false;
  }
}

bool Integral::RunImpl() {
  try {
    if (dimensions_ > 1) {
      if (!func_) {
        throw std::runtime_error("function is not set.");
      }

      std::vector<double> point(dimensions_);
      result_ = Compute(func_, down_limits_, up_limits_, counts_, dimensions_, point, 0);
    } else {
      double step = (up_limits_[0] - down_limits_[0]) / counts_[0];
      result_ = 0.0;
      for (int i = 0; i < counts_[0]; ++i) {
        double x = down_limits_[0] + ((i + 0.5) * step);
        result_ += func_(std::vector<double>{x}) * step;
      }
    }
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error in RunImpl: " << e.what() << '\n';
    return false;
  }
}

bool Integral::PostProcessingImpl() {
  try {
    if (!task_data_ || task_data_->outputs.empty() || task_data_->outputs[0] == nullptr) {
      throw std::invalid_argument("Invalid output data.");
    }
    auto* outputs = reinterpret_cast<double*>(task_data_->outputs[0]);
    outputs[0] = result_;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error in PostProcessingImpl: " << e.what() << '\n';
    return false;
  }
}

void Integral::SetFunction(const std::function<double(double)>& func) {
  func_ = [func](const std::vector<double>& point) { return func(point[0]); };
  dimensions_ = 1;
}

void Integral::SetFunction(const std::function<double(const std::vector<double>&)>& func, int dimensions) {
  func_ = func;
  dimensions_ = dimensions;
  down_limits_.resize(dimensions_);
  up_limits_.resize(dimensions_);
  counts_.resize(dimensions_);
}

double Integral::Compute(const std::function<double(const std::vector<double>&)>& f, const std::vector<double>& a,
                         const std::vector<double>& b, const std::vector<int>& n, int dim, std::vector<double>& point,
                         int current_dim) {
  if (current_dim == dim) {
    return f(point);
  }

  double step = (b[current_dim] - a[current_dim]) / n[current_dim];
  double area = 0.0;

  for (int i = 0; i < n[current_dim]; ++i) {
    point[current_dim] = a[current_dim] + ((i + 0.5) * step);
    area += Compute(f, a, b, n, dim, point, current_dim + 1) * step;
  }

  return area;
}
}  // namespace shurigin_s_integrals_square_seq