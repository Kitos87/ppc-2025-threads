#include "seq/karaseva_e_congrad/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

bool karaseva_e_congrad_seq::TestTaskSequential::PreProcessingImpl() {
  // Set the system size based on the length of vector b
  size_ = task_data->inputs_count[1];
  auto* a_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* b_ptr = reinterpret_cast<double*>(task_data->inputs[1]);

  // Initialize matrix A, vector b and initial guess x (all zeros)
  A_ = std::vector<double>(a_ptr, a_ptr + (size_ * size_));
  b_ = std::vector<double>(b_ptr, b_ptr + size_);
  x_ = std::vector<double>(size_, 0.0);  // Initial guess

  return true;
}

bool karaseva_e_congrad_seq::TestTaskSequential::ValidationImpl() {
  // Check that A is a square matrix (n*n), and b and x have n elements
  const bool valid_input = task_data->inputs_count[0] == task_data->inputs_count[1] * task_data->inputs_count[1];
  const bool valid_output = task_data->outputs_count[0] == task_data->inputs_count[1];
  return valid_input && valid_output;
}

bool karaseva_e_congrad_seq::TestTaskSequential::RunImpl() {
  std::vector<double> r(size_);
  std::vector<double> p(size_);
  std::vector<double> ap(size_);

  // Compute initial residual: r = b - A*x (x is initially zero)
  for (size_t i = 0; i < size_; ++i) {
    r[i] = b_[i];  // x is zero so A*x is zero
    p[i] = r[i];
  }

  double rs_old = 0.0;
  for (size_t i = 0; i < size_; ++i) {
    rs_old += r[i] * r[i];
  }

  const double tolerance = 1e-10;
  const size_t max_iterations = size_;  // Maximum iterations to prevent infinite loops

  for (size_t k = 0; k < max_iterations; ++k) {
    // Compute ap = A * p
    for (size_t i = 0; i < size_; ++i) {
      ap[i] = 0.0;
      for (size_t j = 0; j < size_; ++j) {
        ap[i] += A_[(i * size_) + j] * p[j];
      }
    }

    // Compute alpha = rs_old / (p^T * ap)
    double p_ap = 0.0;
    for (size_t i = 0; i < size_; ++i) {
      p_ap += p[i] * ap[i];
    }
    if (std::fabs(p_ap) < 1e-15) {
      break;  // Avoid division by zero
    }
    const double alpha = rs_old / p_ap;

    // Update solution vector x and residual r
    for (size_t i = 0; i < size_; ++i) {
      x_[i] += alpha * p[i];
      r[i] -= alpha * ap[i];
    }

    // Check convergence by computing the norm of the residual
    double rs_new = 0.0;
    for (size_t i = 0; i < size_; ++i) {
      rs_new += r[i] * r[i];
    }
    if (rs_new < tolerance * tolerance) {  // Compare squared norm to avoid sqrt
      break;
    }

    // Compute beta and update the search direction p
    const double beta = rs_new / rs_old;
    for (size_t i = 0; i < size_; ++i) {
      p[i] = r[i] + beta * p[i];
    }

    rs_old = rs_new;
  }

  return true;
}

bool karaseva_e_congrad_seq::TestTaskSequential::PostProcessingImpl() {
  auto* x_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  for (size_t i = 0; i < x_.size(); ++i) {
    x_ptr[i] = x_[i];
  }
  return true;
}