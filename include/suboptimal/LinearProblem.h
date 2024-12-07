// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace suboptimal {
class LinearProblem {
 public:
  LinearProblem() = default;

  void maximize(const Eigen::Ref<const Eigen::VectorXd>& objective_coeffs);
  void minimize(const Eigen::Ref<const Eigen::VectorXd>& objective_coeffs);

  void addLessThanConstraint(const Eigen::Ref<const Eigen::VectorXd>& constraint_coeffs, double rhs);
  void addGreaterThanConstraint(const Eigen::Ref<const Eigen::VectorXd>& constraint_coeffs, double rhs);
  void addEqualityConstraint(const Eigen::Ref<const Eigen::VectorXd>& constraint_coeffs, double rhs);

  const std::optional<Eigen::VectorXd>& getObjectiveCoeffs() const { return objective_coeffs; }

  bool isMinimization() const { return is_minimization; }

  void buildConstraints(Eigen::Ref<Eigen::MatrixXd> constraint_matrix,
                        Eigen::Ref<Eigen::VectorXd> constraint_rhs) const;

  bool hasInitialBFS() const { return equality_constraints.empty() && greater_than_constraints.empty(); }

  Eigen::Index numDecisionVars() const { return num_decision_vars; }
  Eigen::Index numConstraints() const { return num_constraints; }
  Eigen::Index numSlackVars() const { return numLessThanConstraints() + numGreaterThanConstraints(); }
  Eigen::Index numArtificialVars() const { return numEqualityConstraints() + numGreaterThanConstraints(); }

  Eigen::Index numEqualityConstraints() const { return static_cast<Eigen::Index>(equality_constraints.size()); }
  Eigen::Index numLessThanConstraints() const { return static_cast<Eigen::Index>(less_than_constraints.size()); }
  Eigen::Index numGreaterThanConstraints() const { return static_cast<Eigen::Index>(greater_than_constraints.size()); }

  std::string objectiveFunctionString() const;
  std::vector<std::string> constraintStrings() const;

 private:
  void addConstraintImpl(const Eigen::Ref<const Eigen::VectorXd>& constraint_coeffs, double rhs, int constraint_type);

  bool is_minimization = false;

  std::optional<Eigen::VectorXd> objective_coeffs = std::nullopt;

  std::vector<Eigen::VectorXd> equality_constraints;
  std::vector<Eigen::VectorXd> less_than_constraints;
  std::vector<Eigen::VectorXd> greater_than_constraints;

  Eigen::Index num_decision_vars = 0;
  Eigen::Index num_constraints = 0;
};
}  // namespace suboptimal
