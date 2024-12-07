// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include "InteriorPointConfig.h"
#include "suboptimal/NonlinearProblem.h"
#include "suboptimal/solvers/SolverExitStatus.h"

namespace suboptimal {
SolverExitStatus solveInteriorPoint(NonlinearProblem& problem, const InteriorPointConfig& config);
}  // namespace suboptimal
