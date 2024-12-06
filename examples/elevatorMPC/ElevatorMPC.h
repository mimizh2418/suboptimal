// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include <suboptimal/LinearProblem.h>
#include <suboptimal/solvers/simplex/Simplex.h>

#include <tuple>

#include <Eigen/Core>

#include "Dynamics.h"

// https://math.mit.edu/classes/18.086/2014/reports/LeiZhou.pdf
class ElevatorMPC {
 public:
  ElevatorMPC(const ElevatorDynamics& dynamics, const Eigen::Vector2d& weights, const int horizon)
      : N(horizon),
        vars_per_ts(3 * num_states + num_inputs),  // p+(k+1), v+(k+1), p-(k+1), v-(k+1), p(k+1), v(k+1), u'(k)
        num_decision_vars(N * vars_per_ts),
        dynamics(dynamics),
        weights(weights) {}

  std::tuple<double, suboptimal::SolverExitStatus> calculate(const Eigen::Vector2d& state,
                                                             const Eigen::Vector2d& reference) const {
    // e = |x_i - r_i| = x+_i - x-_i
    // w_i|x_i - r_i| = (w_i)(x+_i) + (w_i)(x-_i)
    // u' = u - u_min
    // Objective: (w_1)(p+) + (w_2)(v+) + (w_1)(p-) + (w_2)(v-)
    Eigen::VectorXd objective = Eigen::VectorXd::Zero(num_decision_vars);
    for (int ts = 0; ts < N; ts++) {
      objective(Eigen::seqN(ts * vars_per_ts, num_states)) = weights;
      objective(Eigen::seqN(ts * vars_per_ts + num_states, num_states)) = weights;
    }
    auto problem = suboptimal::LinearProblem::minimizationProblem(objective);

    for (int ts = 0; ts < N; ts++) {
      const Eigen::Index x_pos_idx = ts * vars_per_ts;
      const Eigen::Index x_neg_idx = x_pos_idx + num_states;
      const Eigen::Index state_idx = x_pos_idx + 2 * num_states;
      const Eigen::Index input_idx = state_idx + num_states;

      // Dynamics constraints
      // x(k+1) = A_d * x(k) + B_d * u(k)
      // x(k+1) = A_d * x(k) + B_d * (u'(k) + u_min)
      Eigen::VectorXd position_constraint = Eigen::VectorXd::Zero(num_decision_vars);
      Eigen::VectorXd velocity_constraint = Eigen::VectorXd::Zero(num_decision_vars);
      Eigen::VectorXd rhs;
      if (ts == 0) {
        // x(k+1) - B_d * u'(k) = A_d * x(k) + B_d * u_min
        rhs = dynamics.A_d * state + dynamics.B_d * (-dynamics.max_input_V);
        position_constraint(state_idx) = 1;
        position_constraint(input_idx) = -dynamics.B_d(0);

        velocity_constraint(state_idx + 1) = 1;
        velocity_constraint(input_idx) = -dynamics.B_d(1);
      } else {
        const Eigen::Index prev_state_idx = (ts - 1) * vars_per_ts + 2 * num_states;
        // x(k+1) - A_d * x(k) - B_d * u'(k) = B_d * u_min
        rhs = dynamics.B_d * (-dynamics.max_input_V);
        position_constraint(state_idx) = 1;
        position_constraint(prev_state_idx) = -dynamics.A_d(0, 0);
        position_constraint(prev_state_idx + 1) = -dynamics.A_d(0, 1);
        position_constraint(input_idx) = -dynamics.B_d(0);

        velocity_constraint(state_idx + 1) = 1;
        velocity_constraint(prev_state_idx) = -dynamics.A_d(1, 0);
        velocity_constraint(prev_state_idx + 1) = -dynamics.A_d(1, 1);
        velocity_constraint(input_idx) = -dynamics.B_d(1);
      }
      problem.addEqualityConstraint(position_constraint, rhs(0));
      problem.addEqualityConstraint(velocity_constraint, rhs(1));

      // Error constraints
      // x+ - x- = x(k+1) - r
      // x+ - x- - x(k+1) = -r
      Eigen::VectorXd position_err_constraint = Eigen::VectorXd::Zero(num_decision_vars);
      position_err_constraint(x_pos_idx) = 1;
      position_err_constraint(x_neg_idx) = -1;
      position_err_constraint(state_idx) = -1;
      problem.addEqualityConstraint(position_err_constraint, -reference(0));

      Eigen::VectorXd velocity_err_constraint = Eigen::VectorXd::Zero(num_decision_vars);
      velocity_err_constraint(x_pos_idx + 1) = 1;
      velocity_err_constraint(x_neg_idx + 1) = -1;
      velocity_err_constraint(state_idx + 1) = -1;
      problem.addEqualityConstraint(velocity_err_constraint, -reference(1));

      // Input constraints
      Eigen::VectorXd input_constraint = Eigen::VectorXd::Zero(num_decision_vars);
      input_constraint(input_idx) = 1;
      problem.addLessThanConstraint(input_constraint, 2 * dynamics.max_input_V);
    }

    // Solve
    Eigen::VectorXd solution(num_decision_vars);
    double cost;
    auto status = suboptimal::solveSimplex(problem, solution, cost);

    return std::make_tuple(solution(vars_per_ts - 1) - dynamics.max_input_V, status);
  }

 private:
  constexpr static int num_states = 2;
  constexpr static int num_inputs = 1;

  const int N;
  const int vars_per_ts;
  const int num_decision_vars;

  const ElevatorDynamics& dynamics;
  const Eigen::Vector2d& weights;
};
