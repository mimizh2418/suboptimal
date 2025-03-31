// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Sparse>

struct Inertia {
  Eigen::Index positive = 0;
  Eigen::Index negative = 0;
  Eigen::Index zero = 0;

  constexpr Inertia() = default;

  constexpr Inertia(const Eigen::Index positive, const Eigen::Index negative, const Eigen::Index zero)
      : positive{positive}, negative{negative}, zero{zero} {}

  explicit Inertia(const Eigen::Ref<const Eigen::VectorXd>& D) {
    for (const double val : D) {
      if (val > 0) {
        positive++;
      } else if (val < 0) {
        negative++;
      } else {
        zero++;
      }
    }
  }

  bool operator==(const Inertia& other) const = default;
};

class RegularizedLDLT {
 public:
  RegularizedLDLT(const Eigen::Index num_decision_vars, const Eigen::Index num_eq_constraints)
      : num_decision_vars{num_decision_vars}, num_eq_constraints{num_eq_constraints} {}

  Eigen::ComputationInfo info() const { return computation_info; }

  RegularizedLDLT& compute(const Eigen::SparseMatrix<double>& lhs) {
    is_sparse = lhs.nonZeros() < 0.25 * lhs.size();  // TODO check sparsity threshold
    computation_info = is_sparse ? computeSparse(lhs).info() : dense_solver.compute(lhs).info();

    Inertia inertia;

    if (computation_info == Eigen::Success) {
      inertia = is_sparse ? Inertia{sparse_solver.vectorD()} : Inertia{dense_solver.vectorD()};
      if (inertia == ideal_inertia) {
        return *this;
      }
    }

    double delta = old_delta == 0.0 ? 1e-4 : old_delta / 2.0;
    double gamma = 1e-10;

    while (true) {
      if (is_sparse) {
        computation_info = computeSparse(lhs + regularization(delta, gamma)).info();
        if (computation_info == Eigen::Success) {
          inertia = Inertia{sparse_solver.vectorD()};
        }
      } else {
        computation_info = dense_solver.compute(lhs + regularization(delta, gamma)).info();
        if (computation_info == Eigen::Success) {
          inertia = Inertia{dense_solver.vectorD()};
        }
      }

      if (computation_info == Eigen::Success) {
        if (inertia == ideal_inertia) {
          old_delta = delta;
          return *this;
        }
        if (inertia.zero > 0) {
          delta *= 10.0;
          gamma *= 10.0;
        } else if (inertia.negative > ideal_inertia.negative) {
          delta *= 10.0;
        } else if (inertia.positive > ideal_inertia.positive) {
          gamma *= 10.0;
        }
      } else {
        delta *= 10.0;
        gamma *= 10.0;
      }

      if (delta > 1e20 || gamma > 1e20) {
        computation_info = Eigen::NumericalIssue;
        return *this;
      }
    }
  }

  template <typename Derived>
  Eigen::VectorXd solve(const Eigen::MatrixBase<Derived>& rhs) {
    if (is_sparse) {
      return sparse_solver.solve(rhs);
    }
    return dense_solver.solve(rhs);
  }

  template <typename Derived>
  Eigen::VectorXd solve(const Eigen::SparseMatrixBase<Derived>& rhs) {
    if (is_sparse) {
      return sparse_solver.solve(rhs);
    }
    return dense_solver.solve(rhs.toDense());
  }

 private:
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> sparse_solver;
  Eigen::LDLT<Eigen::MatrixXd> dense_solver;
  bool is_sparse = true;

  Eigen::ComputationInfo computation_info = Eigen::Success;

  Eigen::Index num_decision_vars = 0;
  Eigen::Index num_eq_constraints = 0;

  Inertia ideal_inertia{num_decision_vars, num_eq_constraints, 0};

  double old_delta = 0.0;
  int non_zeros = -1;

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>& computeSparse(const Eigen::SparseMatrix<double>& lhs) {
    const int non_zeros = lhs.nonZeros();
    if (non_zeros != this->non_zeros) {
      sparse_solver.analyzePattern(lhs);
      this->non_zeros = non_zeros;
    }
    sparse_solver.factorize(lhs);

    return sparse_solver;
  }

  Eigen::SparseMatrix<double> regularization(const double delta, const double gamma) {
    Eigen::VectorXd vec{num_decision_vars + num_eq_constraints};
    vec.head(num_decision_vars).setConstant(delta);
    vec.tail(num_eq_constraints).setConstant(-gamma);

    return Eigen::SparseMatrix<double>{vec.asDiagonal()};
  }
};
