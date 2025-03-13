// Copyright (c) 2024 Alvin Zhang.

#pragma once

#ifndef NDEBUG
#include <iostream>

#define SUBOPTIMAL_ASSERT(condition, message)                          \
  do {                                                                 \
    if (!(condition)) {                                                \
      std::cerr << "Assertion `" #condition "` failed:\n" /* NOLINT */ \
                << (message) << "\n";                                  \
      std::abort();                                                    \
    }                                                                  \
  } while (false)

#else
#define SUBOPTIMAL_ASSERT(condition, message) static_cast<void>(0)
#endif
