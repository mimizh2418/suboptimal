// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include "suboptimal/NonlinearProblem.h"
#include "suboptimal/solvers/ExitStatus.h"
#include "suboptimal/solvers/newton/NewtonConfig.h"

namespace suboptimal {
ExitStatus solveNewton(NonlinearProblem& problem, const NewtonConfig& config = {});

}  // namespace suboptimal
