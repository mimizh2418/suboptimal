// Copyright (c) 2024 Alvin Zhang.

#pragma once

#ifndef NDEBUG
#include <iostream>

#define ASSERT(condition, message)                                                                             \
  do {                                                                                                         \
    if (!(condition)) {                                                                                        \
      std::cerr << "Assertion `" #condition "` failed in " << __FILE__ << " on line " << __LINE__ /* NOLINT */ \
                << ":\n"                                                                                       \
                << (message) << std::endl;                                                                     \
      std::abort();                                                                                            \
    }                                                                                                          \
  } while (false)
#else
#define ASSERT(condition, message) static_cast<void>(0)
#endif
