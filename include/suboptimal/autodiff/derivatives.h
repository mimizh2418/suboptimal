// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include <Eigen/Core>

#include "suboptimal/autodiff/Variable.h"

namespace suboptimal {
/**
 * Computes the gradient of a variable with respect to a set of variables
 * @param var the variable to compute the gradient of
 * @param wrt the variables to compute the gradient with respect to
 * @return a vector of variables representing the gradient of var with respect to wrt
 */
VectorXv gradient(const Variable& var, const Eigen::Ref<const VectorXv>& wrt);

/**
 * Computes the derivative of a variable with respect to another variable
 * @param var the variable to compute the derivative of
 * @param wrt the variable to compute the derivative with respect to
 * @return a variable representing the derivative of var with respect to wrt
 */
Variable derivative(const Variable& var, const Variable& wrt);

/**
 * Computes the Jacobian of a vector of variables with respect to another vector of variables
 * @param vars the vector variables to compute the Jacobian of
 * @param wrt the vector of variables to compute the Jacobian with respect to
 * @return an n x m matrix of variables representing the Jacobian of vars with respect to wrt, where n is the length of
 * vars and m is the length of wrt
 */
MatrixXv jacobian(const Eigen::Ref<const VectorXv>& vars, const Eigen::Ref<const VectorXv>& wrt);

/**
 * Computes the Hessian of a variable with respect to a set of variables
 * @param var the variable to compute the Hessian of
 * @param wrt the variables to compute the Hessian with respect to
 * @return an n x n matrix of variables representing the Hessian of var with respect to wrt, where n is the length of
 * wrt
 */
MatrixXv hessian(const Variable& var, const Eigen::Ref<const VectorXv>& wrt);
}  // namespace suboptimal
