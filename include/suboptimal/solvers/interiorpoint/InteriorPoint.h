// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include "suboptimal/NonlinearProblem.h"
#include "suboptimal/solvers/ExitStatus.h"
#include "suboptimal/solvers/interiorpoint/InteriorPointConfig.h"

namespace suboptimal {
ExitStatus solveInteriorPoint(NonlinearProblem& problem, const InteriorPointConfig& config = {});
}  // namespace suboptimal
