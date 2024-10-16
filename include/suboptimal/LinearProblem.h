#pragma once

#include <Eigen/Core>
#include <string>
#include <vector>

namespace suboptimal {
class LinearProblem {
 public:
  static LinearProblem maximize(const Eigen::VectorXd& objective_coeffs);
  LinearProblem& withLessThanConstraint(const Eigen::VectorXd& constraint_coeffs, double rhs);
  LinearProblem& withGreaterThanConstraint(const Eigen::VectorXd& constraint_coeffs, double rhs);
  LinearProblem& withEqualityConstraint(const Eigen::VectorXd& constraint_coeffs, double rhs);

  const Eigen::VectorXd& getObjectiveCoeffs() const;

  void buildConstraints(Eigen::MatrixXd& constraint_matrix, Eigen::VectorXd& constraint_rhs) const;

  std::string objectiveFunctionString() const;
  std::vector<std::string> constraintStrings() const;

 private:
  explicit LinearProblem(const Eigen::VectorXd& objective_coeffs);
  void addConstraintImpl(const Eigen::VectorXd& constraint_coeffs, double rhs, int constraint_type);

  const Eigen::VectorXd objective_coeffs;

  std::vector<Eigen::VectorXd> equality_constraints;
  std::vector<Eigen::VectorXd> less_than_constraints;
  std::vector<Eigen::VectorXd> greater_than_constraints;

  Eigen::Index num_decision_vars;
  Eigen::Index num_constraints = 0;
};
}  // namespace suboptimal
