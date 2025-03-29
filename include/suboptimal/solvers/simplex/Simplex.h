// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include <Eigen/Core>

#include "suboptimal/LinearProblem.h"
#include "suboptimal/solvers/ExitStatus.h"
#include "suboptimal/solvers/simplex/SimplexConfig.h"

namespace suboptimal {
ExitStatus solveSimplex(const LinearProblem& problem, Eigen::Ref<Eigen::VectorXd> solution, double& objective_value,
                        const SimplexConfig& config = SimplexConfig());
}  // namespace suboptimal
