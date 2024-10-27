// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include <Eigen/Core>

#include "suboptimal/LinearProblem.h"
#include "suboptimal/solvers/SolverExitStatus.h"
#include "suboptimal/solvers/linear/SimplexSolverConfig.h"

namespace suboptimal {
SolverExitStatus solveSimplex(const LinearProblem& problem, Eigen::VectorXd& solution, double& objective_value,
                              const SimplexSolverConfig& config = SimplexSolverConfig());
}  // namespace suboptimal
