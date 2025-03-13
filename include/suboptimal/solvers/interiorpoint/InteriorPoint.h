// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include "suboptimal/NonlinearProblem.h"
#include "suboptimal/solvers/interiorpoint/InteriorPointConfig.h"
#include "suboptimal/solvers/interiorpoint/InteriorPointExitStatus.h"

namespace suboptimal {
InteriorPointExitStatus solveInteriorPoint(NonlinearProblem& problem, const InteriorPointConfig& config = {});
}  // namespace suboptimal
