#pragma once

#include <Eigen/Core>
#include <string>
#include <vector>

namespace suboptimal {
class LinearProblem {
 public:
  static LinearProblem maximizationProblem(const Eigen::VectorXd& objective_coeffs);

  void addLessThanConstraint(const Eigen::VectorXd& constraint_coeffs, double rhs);
  void addGreaterThanConstraint(const Eigen::VectorXd& constraint_coeffs, double rhs);
  void addEqualityConstraint(const Eigen::VectorXd& constraint_coeffs, double rhs);

  const Eigen::VectorXd& getObjectiveCoeffs() const { return objective_coeffs; }

  void buildConstraints(Eigen::MatrixXd& constraint_matrix, Eigen::VectorXd& constraint_rhs) const;

  bool hasInitialBFS() const { return equality_constraints.empty() && greater_than_constraints.empty(); }

  Eigen::Index numDecisionVars() const { return num_decision_vars; }
  Eigen::Index numConstraints() const { return num_constraints; }
  Eigen::Index numSlackVars() const { return numLessThanConstraints() + numGreaterThanConstraints(); }
  Eigen::Index numArtificialVars() const { return numEqualityConstraints() + numGreaterThanConstraints(); }

  Eigen::Index numEqualityConstraints() const;
  Eigen::Index numLessThanConstraints() const;
  Eigen::Index numGreaterThanConstraints() const;

  std::string objectiveFunctionString() const;
  std::vector<std::string> constraintStrings() const;

 private:
  explicit LinearProblem(const Eigen::VectorXd& objective_coeffs);

  void addConstraintImpl(const Eigen::VectorXd& constraint_coeffs, double rhs, int constraint_type);

  Eigen::VectorXd objective_coeffs;

  std::vector<Eigen::VectorXd> equality_constraints;
  std::vector<Eigen::VectorXd> less_than_constraints;
  std::vector<Eigen::VectorXd> greater_than_constraints;

  Eigen::Index num_decision_vars;
  Eigen::Index num_constraints = 0;
};
}  // namespace suboptimal
