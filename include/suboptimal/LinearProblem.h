#pragma once

#include <Eigen/Core>
#include <string>
#include <vector>

namespace suboptimal {
class LinearProblem {
 public:
  static LinearProblem maximize(const Eigen::VectorXd& objective_coeffs);
  LinearProblem& constrainedBy(const Eigen::VectorXd& constraint_coeffs, double rhs);

  [[nodiscard]] const Eigen::VectorXd& getObjectiveCoeffs() const;
  [[nodiscard]] const Eigen::MatrixXd& getConstraintMatrix() const;
  [[nodiscard]] const Eigen::VectorXd& getConstraintRHS() const;

  [[nodiscard]] std::string objectiveFunctionString() const;
  [[nodiscard]] std::vector<std::string> constraintStrings() const;

 private:
  explicit LinearProblem(const Eigen::VectorXd& objective_coeffs);

  Eigen::MatrixXd A;  // constraint matrix
  Eigen::VectorXd b;  // constraint RHS
  Eigen::VectorXd c;  // objective function coefficients
  Eigen::Index num_decision_vars;
  Eigen::Index num_constraints;
};
}  // namespace suboptimal
