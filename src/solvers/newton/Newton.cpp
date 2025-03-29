// Copyright (c) 2024 Alvin Zhang.

#include "suboptimal/solvers/newton/Newton.h"

#include <util/FinalAction.h>

#include <iostream>
#include <limits>

#include <Eigen/Core>
#include <Eigen/SparseCholesky>

#include "suboptimal/autodiff/Derivatives.h"
#include "suboptimal/autodiff/Variable.h"
#include "solvers/RegularizedLDLT.h"
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
    std::cout << std::format("{:^10}{:^10}{:^10}{:^10}\n", "Iteration", "Cost", "||∇f||", "Time (ms)")
              << std::format("{:=<43}", "") << std::endl;
  }

  FinalAction print_diagnostics{[&] {
    if (!config.verbose) {
      return;
    }

    const auto total_time = profiler.totalSolveTime();
    std::cout << "\n"
              << std::format("Solve time: {:.3f} ms ({} iterations; {:.3f} ms average)", total_time.count(),
                             profiler.numIterations(), profiler.avgIterationTime().count())
              << std::endl;
    std::cout << "Exit status: " << toString(exit_status) << std::endl;

    if (exit_status == ExitStatus::Success) {
      std::cout << "Solution: [";
      for (int i = 0; i < x.size(); i++) {
        std::cout << x(i);
        if (i < x.size() - 1) {
          std::cout << ", ";
        }
      }
      std::cout << "]" << std::endl;
    }
    std::cout << std::endl;
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
        std::cout << std::format("{:^10}{:^10.4f}{:^10.4f}{:^10.4f}\n", iterations, f.getValue(), error,
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
