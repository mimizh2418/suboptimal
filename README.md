# suboptimal
subpar optimization algorithm library

C++ optimization library that uses Eigen for linear algebra operations.

Currently only includes an implementation of the 2-phase full tableau simplex method.

## Build
### Dependencies
 - C++20 compiler (only tested with GCC and Clang)
 - [CMake](https://cmake.org/download/) (Might work with CMake 3.12+, only tested with 3.28)
 - [Eigen3](https://gitlab.com/libeigen/eigen)
 - [Catch2](https://github.com/catchorg/Catch2) (tests only)

All missing libraries will be downloaded and build automatically.

## Planned features:
 - Revised simplex method
   - Better optimized simplex implementation that leverages sparse matrix operations
 - SQP method
 - Interior-point method
