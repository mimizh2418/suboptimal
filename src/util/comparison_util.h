// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include <gsl/assert>

template <typename T>
concept ComparableFloat = std::is_floating_point_v<T>;

template <ComparableFloat T>
bool isApprox(T a, T b, T epsilon = std::numeric_limits<T>::epsilon(),
              T abs_thresh = std::numeric_limits<T>::epsilon()) {
  Expects(epsilon >= std::numeric_limits<T>::epsilon() && epsilon <= 1);
  Expects(abs_thresh >= 0);

  if (a == b) {
    return true;
  }
  T norm = std::min(std::abs(a + b), std::numeric_limits<T>::max());
  return std::abs(a - b) <= std::max(abs_thresh, epsilon * norm);
}

template <ComparableFloat T>
bool approxLEQ(T a, T b, T epsilon = std::numeric_limits<T>::epsilon(),
               T abs_thresh = std::numeric_limits<T>::epsilon()) {
  return a < b || isApprox<T>(a, b, epsilon, abs_thresh);
}

template <ComparableFloat T>
bool approxGEQ(T a, T b, T epsilon = std::numeric_limits<T>::epsilon(),
               T abs_thresh = std::numeric_limits<T>::epsilon()) {
  return a > b || isApprox<T>(a, b, epsilon, abs_thresh);
}

template <ComparableFloat T>
bool approxLT(T a, T b, T epsilon = std::numeric_limits<T>::epsilon(),
              T abs_thresh = std::numeric_limits<T>::epsilon()) {
  return !approxGEQ<T>(a, b, epsilon, abs_thresh);
}

template <ComparableFloat T>
bool approxGT(T a, T b, T epsilon = std::numeric_limits<T>::epsilon(),
              T abs_thresh = std::numeric_limits<T>::epsilon()) {
  return !approxLEQ<T>(a, b, epsilon, abs_thresh);
}
