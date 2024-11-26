// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "suboptimal/autodiff/Expression.h"
#include "suboptimal/util/concepts.h"

namespace suboptimal {

/**
 * An autodiff variable. Essentially just a nicer wrapper around Expression
 */
class Variable {
 public:
  /**
   * Constructs an independent variable with initial value 0
   */
  Variable() = default;

  /**
   * Constructs an independent variable with an initial value
   * @param value the value to initialize the variable with
   */
  explicit Variable(double value);

  /**
   * Constructs a dependent variable from an expression
   * @param expr the expression to set this variable to
   */
  Variable(ExpressionPtr&& expr);  // NOLINT

  /**
   * Constructs a dependent variable from an expression
   * @param expr the expression to set this variable to
   */
  Variable(const ExpressionPtr& expr);  // NOLINT

  /**
   * Updates the value of the variable, traversing the expression tree and updating all expressions and variables this
   * variable depends on
   */
  void update();

  /**
   * Sets the value of the variable. No-op if the variable is dependent on other variables/expressions
   * @param value the value to set the variable to
   * @return true if the value was set, false if the variable is dependent
   */
  bool setValue(double value);

  /**
   * Gets the current stored value of the variable. If the variable is dependent and the variables/expressions it
   * depends on may have changed, call update() first to get the correct value
   * @return the value of the variable
   */
  double getValue() const { return expr->value; }

  /**
   * Gets the degree of the expression this variable represents
   * @return the type of the expression
   */
  ExpressionType getType() const { return expr->type; }

  /**
   * Checks if the expression this variable represents is independent of other expressions
   * @return true if the expression is independent
   */
  bool isIndependent() const { return expr->isIndependent(); }

  // Arithmetic operator overloads

  template <typename T>
    requires VariableLike<T>
  Variable& operator+=(const T& other) {
    *this = *this + other;
    return *this;
  }

  template <typename T>
    requires VariableLike<T>
  Variable& operator-=(const T& other) {
    *this = *this - other;
    return *this;
  }

  template <typename T>
    requires VariableLike<T>
  Variable& operator*=(const T& other) {
    *this = *this * other;
    return *this;
  }

  template <typename T>
    requires VariableLike<T>
  Variable& operator/=(const T& other) {
    *this = *this / other;
    return *this;
  }

  friend Variable operator+(const Variable& x);
  friend Variable operator-(const Variable& x);

  template <typename LHS, typename RHS>
    requires VariableLike<LHS> && VariableLike<RHS> && (std::same_as<LHS, Variable> || std::same_as<RHS, Variable>)
  friend Variable operator+(const LHS& lhs, const RHS& rhs);

  template <typename LHS, typename RHS>
    requires VariableLike<LHS> && VariableLike<RHS> && (std::same_as<LHS, Variable> || std::same_as<RHS, Variable>)
  friend Variable operator-(const LHS& lhs, const RHS& rhs);

  template <typename LHS, typename RHS>
    requires VariableLike<LHS> && VariableLike<RHS> && (std::same_as<LHS, Variable> || std::same_as<RHS, Variable>)
  friend Variable operator*(const LHS& lhs, const RHS& rhs);

  template <typename LHS, typename RHS>
    requires VariableLike<LHS> && VariableLike<RHS> && (std::same_as<LHS, Variable> || std::same_as<RHS, Variable>)
  friend Variable operator/(const LHS& lhs, const RHS& rhs);

  friend Variable abs(const Variable& x);
  friend Variable sqrt(const Variable& x);
  friend Variable exp(const Variable& x);
  friend Variable log(const Variable& x);

  template <typename Base, typename Exp>
    requires VariableLike<Base> && VariableLike<Exp> && (std::same_as<Base, Variable> || std::same_as<Exp, Variable>)
  friend Variable pow(const Base& base, const Exp& exponent);

  template <typename X, typename Y>
    requires VariableLike<X> && VariableLike<Y> && (std::same_as<X, Variable> || std::same_as<Y, Variable>)
  friend Variable hypot(const X& x, const Y& y);

  friend Variable sin(const Variable& x);
  friend Variable cos(const Variable& x);
  friend Variable tan(const Variable& x);
  friend Variable asin(const Variable& x);
  friend Variable acos(const Variable& x);
  friend Variable atan(const Variable& x);

  template <typename Y, typename X>
    requires VariableLike<Y> && VariableLike<X> && (std::same_as<Y, Variable> || std::same_as<X, Variable>)
  friend Variable atan2(const Y& y, const X& x);

 private:
  ExpressionPtr expr = std::make_shared<Expression>(0.0, ExpressionType::Linear);
};

Variable operator+(const Variable& x);
Variable operator-(const Variable& x);

template <typename LHS, typename RHS>
  requires VariableLike<LHS> && VariableLike<RHS> && (std::same_as<LHS, Variable> || std::same_as<RHS, Variable>)
Variable operator+(const LHS& lhs, const RHS& rhs) {
  if constexpr (std::is_arithmetic_v<LHS>) {
    return {std::make_shared<Expression>(lhs) + rhs.expr};
  } else if constexpr (std::is_arithmetic_v<RHS>) {
    return {lhs.expr + std::make_shared<Expression>(rhs)};
  } else {
    return {lhs.expr + rhs.expr};
  }
}

template <typename LHS, typename RHS>
  requires VariableLike<LHS> && VariableLike<RHS> && (std::same_as<LHS, Variable> || std::same_as<RHS, Variable>)
Variable operator-(const LHS& lhs, const RHS& rhs) {
  if constexpr (std::is_arithmetic_v<LHS>) {
    return {std::make_shared<Expression>(lhs) - rhs.expr};
  } else if constexpr (std::is_arithmetic_v<RHS>) {
    return {lhs.expr - std::make_shared<Expression>(rhs)};
  } else {
    return {lhs.expr - rhs.expr};
  }
}

template <typename LHS, typename RHS>
  requires VariableLike<LHS> && VariableLike<RHS> && (std::same_as<LHS, Variable> || std::same_as<RHS, Variable>)
Variable operator*(const LHS& lhs, const RHS& rhs) {
  if constexpr (std::is_arithmetic_v<LHS>) {
    return {std::make_shared<Expression>(lhs) * rhs.expr};
  } else if constexpr (std::is_arithmetic_v<RHS>) {
    return {lhs.expr * std::make_shared<Expression>(rhs)};
  } else {
    return {lhs.expr * rhs.expr};
  }
}

template <typename LHS, typename RHS>
  requires VariableLike<LHS> && VariableLike<RHS> && (std::same_as<LHS, Variable> || std::same_as<RHS, Variable>)
Variable operator/(const LHS& lhs, const RHS& rhs) {
  if constexpr (std::is_arithmetic_v<LHS>) {
    return {std::make_shared<Expression>(lhs) / rhs.expr};
  } else if constexpr (std::is_arithmetic_v<RHS>) {
    return {lhs.expr / std::make_shared<Expression>(rhs)};
  } else {
    return {lhs.expr / rhs.expr};
  }
}

Variable abs(const Variable& x);
Variable sqrt(const Variable& x);
Variable exp(const Variable& x);
Variable log(const Variable& x);

template <typename Base, typename Exp>
  requires VariableLike<Base> && VariableLike<Exp> && (std::same_as<Base, Variable> || std::same_as<Exp, Variable>)
Variable pow(const Base& base, const Exp& exponent) {
  if constexpr (std::is_arithmetic_v<Base>) {
    return {pow(std::make_shared<Expression>(base), exponent.expr)};
  } else if constexpr (std::is_arithmetic_v<Exp>) {
    return {pow(base.expr, std::make_shared<Expression>(exponent))};
  } else {
    return {pow(base.expr, exponent.expr)};
  }
}

template <typename X, typename Y>
  requires VariableLike<X> && VariableLike<Y> && (std::same_as<X, Variable> || std::same_as<Y, Variable>)
Variable hypot(const X& x, const Y& y) {
  if constexpr (std::is_arithmetic_v<X>) {
    return {hypot(std::make_shared<Expression>(x), y.expr)};
  } else if constexpr (std::is_arithmetic_v<Y>) {
    return {hypot(x.expr, std::make_shared<Expression>(y))};
  } else {
    return {hypot(x.expr, y.expr)};
  }
}

Variable sin(const Variable& x);
Variable cos(const Variable& x);
Variable tan(const Variable& x);
Variable asin(const Variable& x);
Variable acos(const Variable& x);
Variable atan(const Variable& x);
Variable atan2(const Variable& y, const Variable& x);

template <typename Y, typename X>
  requires VariableLike<Y> && VariableLike<X> && (std::same_as<Y, Variable> || std::same_as<X, Variable>)
Variable atan2(const Y& y, const X& x) {
  if constexpr (std::is_arithmetic_v<Y>) {
    return {atan2(std::make_shared<Expression>(y), x.expr)};
  } else if constexpr (std::is_arithmetic_v<X>) {
    return {atan2(y.expr, std::make_shared<Expression>(x))};
  } else {
    return {atan2(y.expr, x.expr)};
  }
}
}  // namespace suboptimal

namespace Eigen {
template <>
struct NumTraits<suboptimal::Variable> : NumTraits<double> {
  using Real = suboptimal::Variable;
  using NonInteger = suboptimal::Variable;
  using Nested = suboptimal::Variable;

  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 1,
    // TODO check these costs
    ReadCost = 1,
    AddCost = HugeCost,
    MulCost = HugeCost,
  };
};
}  // namespace Eigen

namespace suboptimal {
// Eigen typedefs

using VectorXv = Eigen::VectorX<Variable>;
using Vector2v = Eigen::Vector2<Variable>;
using Vector3v = Eigen::Vector3<Variable>;
using Vector4v = Eigen::Vector4<Variable>;

using MatrixXv = Eigen::MatrixX<Variable>;
using Matrix2v = Eigen::Matrix2<Variable>;
using Matrix3v = Eigen::Matrix3<Variable>;
using Matrix4v = Eigen::Matrix4<Variable>;

/**
 * Returns a matrix holding the values of the Variables in the input matrix
 * @tparam Derived the derived type of the input matrix
 * @param var_mat the input matrix of Variables
 * @return a double matrix holding the values of the Variables in the input matrix
 */
template <typename Derived>
  requires std::same_as<typename Derived::Scalar, Variable>
Eigen::Matrix<double, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> getValues(
    const Eigen::MatrixBase<Derived>& var_mat) {
  Eigen::Matrix<double, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> result(var_mat.rows(), var_mat.cols());
  for (Eigen::Index i = 0; i < var_mat.rows(); i++) {
    for (Eigen::Index j = 0; j < var_mat.cols(); j++) {
      result(i, j) = var_mat(i, j).getValue();
    }
  }
  return result;
}


/**
* Returns a sparse matrix holding the values of the Variables in the sparse input matrix
* @tparam Scalar the scalar type of the input matrix
* @tparam Options the storage scheme of the input matrix
* @tparam StorageIndex the storage index type of the input matrix
* @param var_mat the sparse matrix of Variables
* @return a sparse double matrix holding the values of the Variables in the input matrix
*/
template <typename Scalar, int Options, typename StorageIndex>
  requires std::same_as<Scalar, Variable>
Eigen::SparseMatrix<double> getValues(const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& var_mat) {
  using InputMatType = Eigen::SparseMatrix<Scalar, Options, StorageIndex>;
  using OutputMatType = Eigen::SparseMatrix<double, Options, StorageIndex>;

  OutputMatType result(var_mat.rows(), var_mat.cols());
  std::vector<Eigen::Triplet<double>> triplets{};
  triplets.reserve(var_mat.nonZeros());
  for (Eigen::Index i = 0; i < var_mat.outerSize(); i++) {
    for (typename InputMatType::InnerIterator it(var_mat, i); it; ++it) {
      triplets.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value().getValue()));
    }
  }
  result.setFromTriplets(triplets.begin(), triplets.end());
  return result;
}
}  // namespace suboptimal
