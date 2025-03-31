// Copyright (c) 2024 Alvin Zhang.

#include "suboptimal/solvers/newton/Newton.h"

#include <util/FinalAction.h>

#include <limits>
#include <print>

#include <Eigen/Core>
#include <Eigen/SparseCholesky>

#include "solvers/RegularizedLDLT.h"
#include "suboptimal/autodiff/Derivatives.h"
#include "suboptimal/autodiff/Variable.h"
#include "util/SolverProfiler.h"

using namespace Eigen;

namespace suboptimal {
constexpr double DIV_ITER_THRESH = 1e20;

ExitStatus solveNewton(NonlinearProblem& problem, const NewtonConfig& config) {
  const Variable& f = problem.objectiveFunction();
  VectorXv x_var{problem.decisionVariableVec()};

  Gradient grad_f{f, x_var};
  Hessian hess_f{f, x_var};
  VectorXd x = getValues(x_var);

  SparseVector<double> g = grad_f.getValue();
  SparseMatrix<double> H = hess_f.getValue();

  RegularizedLDLT solver{x_var.size(), 0};

  double error = std::numeric_limits<double>::infinity();

  SolverProfiler profiler{};
  auto exit_status = ExitStatus::MaxIterationsExceeded;

  if (config.verbose) {
    std::println("{:^6} {:^11} {:^11} {:^10}\n{:=<41}", "Iter", "Cost", "||∇f||", "Time (ms)", "");
  }

  FinalAction print_diagnostics{[&] {
    if (!config.verbose) {
      return;
    }

    const auto total_time = profiler.totalSolveTime();
    std::println("\nSolve time: {:.3f} ms ({} iterations; {:.3f} ms average)\nExit status: {}\n", total_time.count(),
                 profiler.numIterations(), profiler.avgIterationTime().count(), toString(exit_status));
  }};

  for (int iterations = 0; iterations < config.max_iterations; iterations++) {
    if (profiler.totalSolveTime() >= config.timeout) {
      exit_status = ExitStatus::Timeout;
      return exit_status;
    }

    profiler.startIteration();
    FinalAction end_iter{[&] {
      profiler.endIteration();

      if (config.verbose) {
        std::println("{:^6} {:^11.3e} {:^11.3e} {:^10.4f}", iterations, f.getValue(), error,
                     profiler.lastIterationTime().count());
      }
    }};

    if (x.lpNorm<Infinity>() > DIV_ITER_THRESH || !x.allFinite()) {
      exit_status = ExitStatus::DivergingIterates;
      return exit_status;
    }

    solver.compute(H);
    VectorXd step = solver.solve(-g);
    x += step;

    setValues(x_var, x);
    g = grad_f.getValue();
    H = hess_f.getValue();

    error = g.norm();

    if (error < config.tolerance) {
      exit_status = ExitStatus::Success;
      return exit_status;
    }
  }

  return exit_status;
}
}  // namespace suboptimal
