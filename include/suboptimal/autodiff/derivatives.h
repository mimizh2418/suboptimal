// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include <vector>

#include <Eigen/Core>

#include "suboptimal/autodiff/Variable.h"

namespace suboptimal {
/**
 * Class for computing and storing the gradient of a variable. Caches results of linear and constant expressions and
 * only recomputes quadratic and nonlinear expressions.
 */
class Gradient {
 public:
  /**
   * Constructs a gradient object
   * @param var the variable to compute the gradient of
   * @param wrt the vector of variables to compute the gradient with respect to
   */
  Gradient(const Variable& var, const Eigen::Ref<const VectorXv>& wrt);

  /**
   * Gets the value of the gradient based on the current value of wrt
   */
  const Eigen::SparseVector<double>& getValue();

  /**
   * Gets a vector of variables representing the gradient algebraically
   */
  VectorXv getExpr();

 private:
  Variable var;
  VectorXv wrt;
  Eigen::SparseVector<double> value;

  friend class Jacobian;
};

/**
 * Class for computing and storing the derivative of a variable with respect to another variable. Caches results of
 * linear and constant expressions and only recomputes quadratic and nonlinear expressions.
 */
class Derivative {
 public:
  /**
   * Constructs a Derivative object
   * @param var the variable to compute the derivative of
   * @param wrt the variable to compute the derivative with respect to
   */
  Derivative(const Variable& var, const Variable& wrt);

  /**
   * Gets the value of the derivative based on the current value of wrt
   */
  double getValue();

  /**
   * Gets a variable representing the derivative algebraically
   */
  Variable getExpr();

 private:
  Variable var;
  Variable wrt;
  Gradient gradient;
};

/**
 * Class for computing and storing the Jacobian of a vector of variables. Caches results of linear and constant
 * expressions and only recomputes quadratic and nonlinear expressions.
 */
class Jacobian {
 public:
  /**
   * Constructs a Jacobian object
   * @param vars the variable to compute the gradient of
   * @param wrt the vector of variables to compute the gradient with respect to
   */
  Jacobian(const Eigen::Ref<const VectorXv>& vars, const Eigen::Ref<const VectorXv>& wrt);

  /**
   * Gets the value of the Jacobian based on the current value of wrt
   */
  const Eigen::SparseMatrix<double>& getValue();

  /**
   * Gets a matrix of variables representing the Jacobian algebraically
   */
  MatrixXv getExpr();

 private:
  VectorXv vars;
  VectorXv wrt;
  Eigen::SparseMatrix<double> value;

  std::vector<Gradient> gradients;
  std::vector<int> nonlinear_rows;
  std::vector<Eigen::Triplet<double>> cache;
};

/**
 * Class for computing and storing the Hessian of a variable. Caches results of linear and constant expressions and
 * only recomputes quadratic and nonlinear expressions.
 */
class Hessian {
 public:
  /**
   * Constructs a Hessian object
   * @param var the variable to compute the Hessian of
   * @param wrt the vector of variables to compute the Hessian with respect to
   */
  Hessian(const Variable& var, const Eigen::Ref<const VectorXv>& wrt);

  /**
   * Gets the value of the Hessian based on the current value of wrt
   */
  const Eigen::SparseMatrix<double>& getValue();

  /**
   * Gets a matrix of variables representing the Hessian algebraically
   */
  MatrixXv getExpr();

 private:
  Jacobian jacobian;
};
}  // namespace suboptimal
