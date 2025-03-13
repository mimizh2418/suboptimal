// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include <Eigen/Core>

#include "suboptimal/LinearProblem.h"
#include "suboptimal/solvers/simplex/SimplexConfig.h"
#include "suboptimal/solvers/simplex/SimplexExitStatus.h"

namespace suboptimal {
SimplexExitStatus solveSimplex(const LinearProblem& problem, Eigen::Ref<Eigen::VectorXd> solution,
                               double& objective_value, const SimplexConfig& config = SimplexConfig());
}  // namespace suboptimal
