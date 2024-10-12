#pragma once

#include <Eigen/Core>

namespace suboptimal {
class SimplexProblem {
 public:
  SimplexProblem();
  SimplexProblem& maximize(const Eigen::VectorXd& objective_coeffs);
  SimplexProblem& constrainedBy(const Eigen::VectorXd& constraint_coeffs, double rhs);
  double solve(Eigen::VectorXd& solution);
  const Eigen::MatrixXd& getTableau();

 private:
  Eigen::MatrixXd A;  // constraint matrix
  Eigen::VectorXd b;  // constraint RHS
  Eigen::VectorXd c;  // objective function coefficients
  Eigen::Index num_decision_vars;
  Eigen::Index num_constraints;
  Eigen::MatrixXd tableau;

  void initTableau();
  bool findPivotPosition(Eigen::Index& pivot_row, Eigen::Index& pivot_col);
  Eigen::Index getBasicRow(Eigen::Index col);
  void pivot(Eigen::Index pivot_row, Eigen::Index pivot_col);
};
}  // namespace suboptimal
