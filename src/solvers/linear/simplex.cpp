// Copyright (c) 2024 Alvin Zhang.

#include "suboptimal/solvers/linear/simplex.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#include <Eigen/Core>
#include <gsl/util>

#include "suboptimal/LinearProblem.h"
#include "suboptimal/solvers/SolverExitStatus.h"
#include "suboptimal/solvers/linear/SimplexPivotRule.h"
#include "suboptimal/solvers/linear/SimplexSolverConfig.h"
#include "util/SolverProfiler.h"
#include "util/comparison_util.h"

using namespace Eigen;

namespace suboptimal {
int findPivotPosition(const MatrixXd& tableau, const VectorX<Index>& basic_vars, const SimplexPivotRule pivot_rule,
                      Index& pivot_row, Index& pivot_col) {
  pivot_col = -1;
  pivot_row = -1;

  // Entering variable selection
  // TODO: is there a less scuffed typedef for this?
  const Block objective_row = tableau.row(tableau.rows() - 1).head(tableau.cols() - 1);
  if (pivot_rule == SimplexPivotRule::Bland) {
    // Bland's rule: select the index of the first negative coefficient in the objective row
    const auto it = std::ranges::find_if(objective_row, [](const double coeff) { return approxLT<double>(coeff, 0); });
    if (it != objective_row.end()) {
      pivot_col = it - objective_row.begin();
    }
  } else if (pivot_rule == SimplexPivotRule::Lexicographic || pivot_rule == SimplexPivotRule::Dantzig) {
    // Dantzig's or lexicographic rule: select the index of the most negative coefficient in the objective row
    Index smallest_idx;
    const double smallest_coeff = objective_row.minCoeff(&smallest_idx);
    if (approxLT<double>(smallest_coeff, 0)) {
      pivot_col = smallest_idx;
    }
  }

  // If no negative coefficients are found, the solution is optimal
  if (pivot_col == -1) {
    return 1;
  }

  // Leaving variable selection
  double min_ratio = std::numeric_limits<double>::infinity();
  const MatrixXd::ConstColXpr pivot_col_coeffs = tableau.col(pivot_col);
  const MatrixXd::ConstColXpr rhs_coeffs = tableau.col(tableau.cols() - 1);
  for (Index i = 0; i < tableau.rows() - 1; i++) {
    const double ratio = rhs_coeffs(i) / pivot_col_coeffs(i);
    // Pivot element cannot be 0 and ratio cannot be negative
    if (approxLEQ<double>(tableau(i, pivot_col), 0) || approxLT<double>(ratio, 0)) {
      continue;
    }
    // Minimum ratio test
    if (approxLT<double>(ratio, min_ratio)) {
      min_ratio = ratio;
      pivot_row = i;
      continue;
    }
    // Dantzig does not have tie-breaking between equal ratios
    if (!isApprox<double>(ratio, min_ratio) || pivot_rule == SimplexPivotRule::Dantzig) {
      continue;
    }
    if (pivot_rule == SimplexPivotRule::Lexicographic) {
      // Lexicographic rule: select the variable with the lexicographically smallest row
      const RowVectorXd row_candidate = tableau.row(i) / tableau(i, pivot_col);
      const RowVectorXd current_best_row = tableau.row(pivot_row) / tableau(pivot_row, pivot_col);
      if (std::ranges::lexicographical_compare(row_candidate, current_best_row,
                                               [](const double a, const double b) { return approxLT<double>(a, b); })) {
        pivot_row = i;
      }
    } else {
      // Bland's rule: select the basic variable with the smallest index
      if (basic_vars(i) < basic_vars(pivot_row)) {
        pivot_row = i;
      }
    }
  }

  // If no valid pivot is found, the problem is unbounded
  if (pivot_row == -1) {
    return -1;
  }

  // Valid pivot found
  return 0;
}

void pivot(MatrixXd& tableau, VectorX<Index>& basic_vars, const Index pivot_row, const Index pivot_col) {
  const double pivot_element = tableau(pivot_row, pivot_col);
  tableau.row(pivot_row) /= pivot_element;
  tableau(pivot_row, pivot_col) = 1;  // Account for floating point errors
  for (Index i = 0; i < tableau.rows(); i++) {
    if (i == pivot_row || isApprox<double>(tableau(i, pivot_col), 0)) {
      continue;
    }
    tableau.row(i) -= tableau(i, pivot_col) * tableau.row(pivot_row);
    tableau(i, pivot_col) = 0;  // Account for floating point errors
  }
  // Update basic variables
  basic_vars(pivot_row) = pivot_col;
}

VectorX<Index> findBasicVars(const MatrixXd& tableau) {
  VectorX<Index> basic_vars = VectorX<Index>::Zero(tableau.rows() - 1);
  for (Index i = 0; i < tableau.cols(); i++) {
    const MatrixXd::ConstColXpr col = tableau.col(i);
    Index max_index;
    if (isApprox<double>(col.lpNorm<1>(), 1) && isApprox<double>(col.maxCoeff(&max_index), 1)) {
      basic_vars(max_index) = i;
    }
  }
  return basic_vars;
}

SolverExitStatus solveTableau(MatrixXd& tableau, VectorX<Index>& basic_vars, SolverProfiler& profiler,
                              const SimplexSolverConfig& config) {
  for (int i = 0; i < config.max_iterations; i++) {
    // Check for timeout
    if (profiler.totalSolveTime() >= config.timeout) {
      return SolverExitStatus::Timeout;
    }

    profiler.startIteration();
    auto end_profile_iter = gsl::finally([&] { profiler.endIteration(); });

    // Find pivot position
    Index pivot_row, pivot_col;
    const int pivot_status = findPivotPosition(tableau, basic_vars, config.pivot_rule, pivot_row, pivot_col);
    if (pivot_status == -1) {
      return SolverExitStatus::Unbounded;  // Could not find a valid pivot position, problem is unbounded
    }
    if (pivot_status == 1) {
      return SolverExitStatus::Success;  // Optimal solution found
    }
    // Perform pivot operation
    pivot(tableau, basic_vars, pivot_row, pivot_col);
  }

  return SolverExitStatus::MaxIterationsExceeded;
}

SolverExitStatus solveSimplex(const LinearProblem& problem, Ref<VectorXd> solution, double& objective_value,
                              const SimplexSolverConfig& config) {
  Expects(solution.size() == problem.numDecisionVars());

  Index num_vars = problem.numDecisionVars() + problem.numSlackVars() + problem.numArtificialVars();

  // Initialize tableau
  MatrixXd tableau = MatrixXd::Zero(problem.numConstraints() + 1, num_vars + 1);
  problem.buildConstraints(tableau.topLeftCorner(problem.numConstraints(), num_vars),
                           tableau.topRightCorner(problem.numConstraints(), 1));
  tableau.bottomLeftCorner(1, problem.numDecisionVars()) = -problem.getObjectiveCoeffs().transpose();

  if (config.verbose) {
    std::cout << "Solving linear problem \n"
              << (problem.isMinimization() ? "Minimize: " : "Maximize: ") << problem.objectiveFunctionString() << "\n"
              << "Subject to: " << "\n";
    for (const auto constraint_strings = problem.constraintStrings();
         const auto& constraint_string : constraint_strings) {
      std::cout << "  " << constraint_string << "\n";
    }
    std::cout << "Using pivot rule: " << toString(config.pivot_rule) << "\n";
    std::cout << std::endl;
  }

  if (problem.numConstraints() == 0) {
    std::cout << "Solver failed to find a solution: " << toString(SolverExitStatus::Unbounded) << std::endl;
    return SolverExitStatus::Unbounded;
  }

  VectorX<Index> basic_vars;
  if (!problem.hasInitialBFS()) {
    if (config.verbose) {
      std::cout << "No trivial BFS found, solving auxiliary LP" << std::endl;
    }

    // Set up auxiliary LP
    const Index num_artificial_vars = problem.numEqualityConstraints() + problem.numGreaterThanConstraints();

    // Construct auxiliary objective function
    RowVectorXd auxiliary_objective = RowVectorXd::Zero(tableau.cols());
    auxiliary_objective.segment(problem.numDecisionVars() + problem.numSlackVars(), num_artificial_vars) =
        RowVectorXd::Ones(num_artificial_vars);

    // Subtract rows with artificial variables from the auxiliary objective to make it a valid objective function
    const MatrixXd::BlockXpr artificial_rows =
        tableau.middleRows(problem.numLessThanConstraints(), num_artificial_vars);
    RowVectorXd artificial_row_sum = RowVectorXd::Zero(tableau.cols());
    for (auto row : artificial_rows.rowwise()) {
      artificial_row_sum += row;
    }
    auxiliary_objective -= artificial_row_sum;

    // Replace the original objective function with the auxiliary objective
    tableau.row(tableau.rows() - 1) = auxiliary_objective;

    // Find basic variables
    basic_vars = findBasicVars(tableau);

    // Perform simplex iterations to find initial BFS
    SolverProfiler aux_profiler{};
    const SolverExitStatus aux_exit = solveTableau(tableau, basic_vars, aux_profiler, config);
    const bool is_feasible = isApprox<double>(tableau(tableau.rows() - 1, tableau.cols() - 1), 0);

    auto print_diagnostics = gsl::finally([&] {
      if (!config.verbose) {
        return;
      }

      const auto total_time = aux_profiler.totalSolveTime();
      std::cout << std::format("Auxiliary LP solve time: {:.3f} ms ({} iterations; {:.3f} ms average)",
                               total_time.count(), aux_profiler.numIterations(),
                               aux_profiler.avgIterationTime().count())
                << "\n";

      if (aux_exit != SolverExitStatus::Success) {
        std::cout << "Solving auxiliary LP failed: " << toString(aux_exit) << "\n";
      } else if (!is_feasible) {
        std::cout << "Solver failed to find a solution: " << toString(SolverExitStatus::Infeasible) << "\n";
      }
      std::cout << std::endl;
    });

    if (aux_exit != SolverExitStatus::Success) {
      return aux_exit;
    }

    if (!isApprox<double>(tableau(tableau.rows() - 1, tableau.cols() - 1), 0)) {
      return SolverExitStatus::Infeasible;
    }

    // Remove non-basic artificial variables from tableau
    std::vector<Index> cols_to_keep(problem.numDecisionVars() + problem.numSlackVars());
    std::iota(cols_to_keep.begin(), cols_to_keep.end(), 0);
    for (Index i = problem.numDecisionVars() + problem.numSlackVars(); i < tableau.cols() - 1; i++) {
      if ((basic_vars.array() == i).any()) {
        cols_to_keep.push_back(i);
      }
    }
    cols_to_keep.push_back(tableau.cols() - 1);  // Keep the RHS
    tableau = tableau(all, cols_to_keep).eval();
    basic_vars = findBasicVars(tableau);  // Recalculate basic variables on new tableau

    RowVectorXd objective_row = RowVectorXd::Zero(tableau.cols());
    objective_row.head(problem.numDecisionVars()) = -problem.getObjectiveCoeffs().transpose();
    for (Index i = 0; i < basic_vars.size(); i++) {
      objective_row -= tableau.row(i) * objective_row(basic_vars(i));
    }
    tableau.row(tableau.rows() - 1) = objective_row;
  } else {
    basic_vars = findBasicVars(tableau);
  }

  // Perform simplex iterations
  SolverProfiler profiler{};
  const SolverExitStatus exit_status = solveTableau(tableau, basic_vars, profiler, config);

  // TODO why is this here
  auto print_diagnostics = gsl::finally([&] {
    if (!config.verbose) {
      return;
    }

    const auto total_time = profiler.totalSolveTime();
    std::cout << std::format("Solve time: {:.3f} ms ({} iterations; {:.3f} ms average)", total_time.count(),
                             profiler.numIterations(), profiler.avgIterationTime().count())
              << "\n";

    if (exit_status != SolverExitStatus::Success) {
      std::cout << "Solver failed to find a solution: " << toString(exit_status) << "\n" << std::endl;
      return;
    }

    std::cout << "Solution:\n";
    for (Index i = 0; i < problem.numDecisionVars(); i++) {
      std::cout << "  x_" << i + 1 << " = " << solution(i) << "\n";
    }
    std::cout << "Objective value: " << objective_value << "\n" << std::endl;
  });

  if (exit_status != SolverExitStatus::Success) {
    return exit_status;
  }

  // Extract solution and objective value
  solution = VectorXd::Zero(problem.numDecisionVars());
  const MatrixXd::ColXpr rhs = tableau.col(tableau.cols() - 1);
  for (Index i = 0; i < basic_vars.size(); i++) {
    if (const Index var_index = basic_vars(i); var_index < problem.numDecisionVars()) {
      solution(var_index) = rhs(i);
    }
  }
  objective_value = (problem.isMinimization() ? -1 : 1) * tableau(tableau.rows() - 1, tableau.cols() - 1);

  return exit_status;
}
}  // namespace suboptimal
