#pragma once

#include <cmath>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sidorina_p_gradient_method_seq {

inline std::vector<double> MultiplyMatrixByVector(const std::vector<double>& a, const std::vector<double>& vec,
                                                  int size) {
  std::vector<double> result(size, 0);
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      result[i] += a[(i * size) + j] * vec[j];
    }
  }
  return result;
}

inline double VectorNorm(const std::vector<double>& vec) {
  double sum = 0;
  for (double value : vec) {
    sum += std::pow(value, 2);
  }
  return std::sqrt(sum);
}

inline double Dot(const std::vector<double>& vec1, const std::vector<double>& vec2) {
  double sum = 0;
  for (unsigned long i = 0; i < vec1.size(); i++) {
    sum += vec1[i] * vec2[i];
  }
  return sum;
}

inline double Dot(const std::vector<double>& vec) {
  double sum = 0;
  for (unsigned long i = 0; i < vec.size(); i++) {
    sum += std::pow(vec[i], 2);
  }
  return sum;
}

inline std::vector<double> ConjugateGradientMethod(std::vector<double>& a, std::vector<double>& b,
                                                   std::vector<double> solution, double tolerance, int size) {
  std::vector<double> matrix_times_solution = MultiplyMatrixByVector(a, solution, size);

  auto residual = std::vector<double>(size);
  auto direction = std::vector<double>(size);

  for (int i = 0; i < size; ++i) {
    residual[i] = b[i] - matrix_times_solution[i];
  }

  double residual_norm_squared = Dot(residual);
  if (std::sqrt(residual_norm_squared) < tolerance) {
    return solution;
  }
  direction = residual;
  std::vector<double> matrix_times_direction(size);
  while (std::sqrt(residual_norm_squared) > tolerance) {
    matrix_times_direction = MultiplyMatrixByVector(a, direction, size);
    double direction_dot_matrix_times_direction = Dot(direction, matrix_times_direction);
    double alpha = residual_norm_squared / direction_dot_matrix_times_direction;
    for (int i = 0; i < size; i++) {
      solution[i] += alpha * direction[i];
    }
    for (int i = 0; i < size; i++) {
      residual[i] -= alpha * matrix_times_direction[i];
    }

    double new_residual_norm_squared = Dot(residual);
    double beta = new_residual_norm_squared / residual_norm_squared;
    residual_norm_squared = new_residual_norm_squared;
    for (int i = 0; i < size; i++) {
      direction[i] = residual[i] + beta * direction[i];
    }
  }
  return solution;
}

inline bool Cholesky(const std::vector<double>& matrix, int w, int h, double tolerance = 1e-5) {
  if (w != h) {
    return false;
  }

  int n = w;

  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      if (std::abs(matrix[(i * n) + j] - matrix[(j * n) + i]) > tolerance) {
        return false;
      }
    }
  }

  std::vector<double> lower_triangular(n * n, 0.0);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
      double sum = 0.0;
      for (int k = 0; k < j; k++) {
        sum += lower_triangular[(i * n) + k] * lower_triangular[(j * n) + k];
      }

      if (i == j) {
        double diag_val = matrix[(i * n) + i] - sum;
        if (diag_val <= tolerance) {
          return false;
        }
        lower_triangular[(i * n) + i] = std::sqrt(diag_val);
      } else {
        lower_triangular[(i * n) + j] = (1.0 / lower_triangular[(j * n) + j]) * (matrix[(i * n) + j] - sum);
      }
    }
  }

  return true;
}

class GradientMethod : public ppc::core::Task {
 public:
  explicit GradientMethod(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int size_;
  double tolerance_;
  std::vector<double> a_;
  std::vector<double> b_;
  std::vector<double> solution_;
  std::vector<double> result_;
};

}  // namespace sidorina_p_gradient_method_seq