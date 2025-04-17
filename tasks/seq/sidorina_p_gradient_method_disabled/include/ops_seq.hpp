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

inline double CalculateDeterminant(const double* a, int size) {
  double d = 0;
  if (size == 1) {
    d = a[0];
  } else if (size == 2) {
    d = (a[0] * a[3]) - (a[1] * a[2]);
  } else {
    double determinant = 0;
    for (int i = 0; i < size; ++i) {
      double cofactor = std::pow(-1, i) * CalculateDeterminant(a + (size * 1), size - 1);
      determinant += a[i] * cofactor;
    }
    d = determinant;
  }
  return d;
}

inline bool MatrixSimmPositive(const double* a, int size) {
  std::vector<double> a0(size * size);
  for (int i = 0; i < size * size; i++) {
    a0[i] = a[i];
  }

  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (a0[(i * size) + j] != a0[(j * size) + i]) {
        return false;
      }
    }
  }

  std::vector<double> minors(size);
  for (int i = 1; i <= size; i++) {
    auto* submatrix = new double[i * i];
    for (int j = 0; j < i; j++) {
      for (int k = 0; k < i; k++) {
        submatrix[(j * i) + k] = a[(j * size) + k];
      }
    }

    minors[i - 1] = CalculateDeterminant(submatrix, i);
    delete[] submatrix;
  }
  for (unsigned long i = 0; i < minors.size(); ++i) {
    if (minors[i] <= 0) {
      return false;
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