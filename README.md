# suboptimal
subpar optimization algorithm library

C++ optimization library that uses Eigen for linear algebra operations.

Currently only includes an implementation of the full-tableau simplex method for only maximization problems with
trivially determinable basic feasible solutions.

## Build
### Dependencies
 - C++20 compiler (only tested with GCC)
 - [CMake](https://cmake.org/download/) (Might work with CMake 3.12+, only tested with 3.28)
 - [Eigen3](https://gitlab.com/libeigen/eigen)
   - Will be downloaded and built automatically if not found on the system

## Planned features:
 - 2-phase simplex method
    - Will add support for minimization problems and more complex constraints
 - Revised simplex method
   - Better optimized simplex implementation that leverages sparse matrix operations
 - Interior-point method
    - Sometime in the not too near future
 - Other methods (like SQP, etc.)
    - Maybe?