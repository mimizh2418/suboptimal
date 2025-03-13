// Copyright (c) 2024 Alvin Zhang.

#include "suboptimal/solvers/interiorpoint/InteriorPoint.h"

#include <algorithm>
#include <limits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCholesky>

#include "suboptimal/autodiff/Derivatives.h"
#include "suboptimal/autodiff/Variable.h"
#include "suboptimal/solvers/interiorpoint/InteriorPointConfig.h"
#include "suboptimal/util/Assert.h"

using namespace Eigen;

// https://www.math.uci.edu/~qnie/Publications/NumericalOptimization.pdf (ch. 19)
namespace suboptimal {
constexpr double DIV_ITER_THRESH = 1e20;    // Threshold for when solver considers iterates to be diverging
constexpr double INFEASIBLE_THRESH = 1e-6;  // Threshold for when solver considers constraints to be infeasible
constexpr double TAU_MIN = 0.99;            // Fraction to the boundary scale factor

inline double errorEstimate(const VectorXd& grad_f, const VectorXd& ce, const SparseMatrix<double>& Ae,
                            const VectorXd& ci, const SparseMatrix<double>& Ai, const VectorXd& y, const VectorXd& z,
                            const VectorXd& s, const double mu) {
  constexpr double s_max = 100.0;
  const double s_d =
      std::max({s_max, (y.lpNorm<1>() + z.lpNorm<1>()) / static_cast<double>(ce.size() + ci.size())}) / s_max;
  const double s_c = std::max({s_max, z.lpNorm<1>() / static_cast<double>(ci.size())}) / s_max;

  const auto S = s.asDiagonal();
  const VectorXd e = VectorXd::Ones(s.size());

  return std::max({(grad_f - Ae.transpose() * y - Ai.transpose() * z).lpNorm<Infinity>() / s_d,
                   (S * z - mu * e).lpNorm<Infinity>() / s_c, ce.lpNorm<Infinity>(), (ci - s).lpNorm<Infinity>()});
}

inline double fractionToTheBoundary(const VectorXd& var, const VectorXd& p, const double tau) {
  SUBOPTIMAL_ASSERT(var.size() == p.size(), "Variable size and iterate size must be same");

  double alpha = 1.0;
  for (Index i = 0; i < var.size(); i++) {
    if (alpha * p(i) < -tau * var(i)) {
      alpha = -tau / p(i) * var(i);
    }
  }

  return alpha;
}

inline bool isLocallyInfeasible(const SparseMatrix<double>& A_e, const VectorXd& c_e, const SparseMatrix<double>& A_i,
                                const VectorXd& c_i) {
  const VectorXd c_i_plus = c_i.cwiseMin(0.0);

  const bool is_equality_infeasible =
      A_e.rows() > 0 && (A_e.transpose() * c_e).norm() < INFEASIBLE_THRESH && c_e.norm() > 1e-2;
  const bool is_inequality_infeasible =
      A_i.rows() > 0 && (A_i.transpose() * c_i_plus).norm() < INFEASIBLE_THRESH && c_i_plus.norm() > 1e-6;
  return is_equality_infeasible || is_inequality_infeasible;
}

InteriorPointExitStatus solveInteriorPoint(NonlinearProblem& problem, const InteriorPointConfig& config) {
  const Variable& f = problem.objectiveFunction();
  VectorXv x_var{static_cast<Index>(problem.decisionVariables().size())};  // Decision variables
  std::ranges::copy(problem.decisionVariables().begin(), problem.decisionVariables().end(), x_var.begin());
  VectorXd x = getValues(x_var);

  VectorXv c_e_var{static_cast<Index>(problem.equalityConstraints().size())};  // Equality constraints
  std::ranges::copy(problem.equalityConstraints().begin(), problem.equalityConstraints().end(), c_e_var.begin());
  VectorXd c_e = getValues(c_e_var);

  VectorXv c_i_var{problem.inequalityConstraints().size()};  // Inequality constraints
  std::ranges::copy(problem.inequalityConstraints().begin(), problem.inequalityConstraints().end(), c_i_var.begin());
  VectorXd c_i = getValues(c_i_var);

  // Slack variables
  VectorXv s_var = VectorXv::Ones(c_e.size());
  VectorXd s = getValues(s_var);

  // Lagrange multipliers
  VectorXv y_var{c_e.size()};
  VectorXd y = getValues(y_var);

  VectorXv z_var{c_i.size()};
  VectorXd z = getValues(z_var);

  Variable L = f - y_var.dot(c_e_var) - z_var.dot(c_i_var - s_var);  // Lagrangian

  // Derivatives
  Jacobian j_ce{c_e_var, x_var};
  SparseMatrix<double> A_e = j_ce.getValue();

  Jacobian j_ci{c_i_var, x_var};
  SparseMatrix<double> A_i = j_ci.getValue();

  Gradient grad_f{f, x_var};
  SparseVector<double> g = grad_f.getValue();

  Hessian hess_L{L, x_var};
  SparseMatrix<double> H = hess_L.getValue();

  // Barrier parameter
  double mu = 0.1;

  double tau = TAU_MIN;

  double estimated_error = std::numeric_limits<double>::infinity();

  for (int iter_count = 0; iter_count < config.max_iterations; iter_count++) {
    if (estimated_error < config.tolerance) {
      return InteriorPointExitStatus::Success;
    }

    if (isLocallyInfeasible(A_e, c_e, A_i, c_i)) {
      return InteriorPointExitStatus::Infeasible;
    }

    // Check for diverging iterates
    if (x.lpNorm<Infinity>() >= DIV_ITER_THRESH || s.lpNorm<Infinity>() >= DIV_ITER_THRESH || !x.allFinite() ||
        !s.allFinite()) {
      return InteriorPointExitStatus::DivergingIterates;
    }

    const auto S = s.asDiagonal();
    const auto Z = z.asDiagonal();
    const SparseMatrix<double> Sigma = SparseMatrix<double>{S.inverse()} * SparseMatrix<double>{Z};

    // Construct and solve the KKT system
    SparseMatrix<double> coeffs{x.size() + c_e.size(), x.size() + c_e.size()};
    SparseMatrix<double> top_left = H + A_i.transpose() * Sigma * A_i;

    std::vector<Triplet<double>> triplets{};
    triplets.reserve(top_left.nonZeros() + 2 * A_e.nonZeros());
    for (Index i = 0; i < top_left.outerSize(); i++) {
      for (SparseMatrix<double>::InnerIterator it(top_left, i); it; ++it) {
        triplets.emplace_back(i, it.col(), it.value());
      }
    }
    for (Index i = 0; i < A_e.outerSize(); i++) {
      for (SparseMatrix<double>::InnerIterator it(A_e, i); it; ++it) {
        triplets.emplace_back(i + x.size(), it.col(), it.value());
        triplets.emplace_back(it.col(), i + x.size(), it.value());
      }
    }
    coeffs.setFromTriplets(triplets.begin(), triplets.end());

    VectorXd rhs{x.size() + y.size()};
    VectorXd e = VectorXd::Ones(s.size());
    rhs.head(x.size()) = -(g - A_e.transpose() * y * A_i.transpose() * (S.inverse() * (Z * c_i - mu * e - z)));
    rhs.tail(y.size()) = c_e;

    SimplicialLDLT<SparseMatrix<double>> solver{};
    VectorXd step{x.size() + y.size()};
    solver.compute(coeffs);
    if (solver.info() != Success) {
      // TODO: handle this case
      SUBOPTIMAL_ASSERT(false, "Solver failed to factorize the KKT matrix");
    } else {
      step = solver.solve(rhs);
    }

    VectorXd p_x = step.head(x.size());
    VectorXd p_y = -step.tail(s.size());
    VectorXd p_z = -Sigma * c_i + mu * S.inverse() * e - Sigma * A_i * p_x;
    VectorXd p_s = c_i - s + A_i * p_x;

    double alpha_max = fractionToTheBoundary(s, p_s, tau);
    double alpha = alpha_max;
    double alpha_z = fractionToTheBoundary(z, p_z, tau);
  }

  return InteriorPointExitStatus::MaxIterationsExceeded;
}
}  // namespace suboptimal
