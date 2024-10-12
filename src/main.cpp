#include <iostream>

#include "suboptimal/solver/simplex.h"

using namespace suboptimal;
using namespace std;
using namespace Eigen;

int main() {
  SimplexProblem problem{};
  VectorXd solution;
  problem.maximize(VectorXd{{10, -57, -9, -24}})
      .constrainedBy(VectorXd{{0.5, -5.5, -2.5, 9}}, 0)
      .constrainedBy(VectorXd{{0.5, -1.5, -0.5, 1}}, 0)
      .constrainedBy(VectorXd{{1, 0, 0, 0}}, 1);
  const double max = problem.solve(solution);
  cout << "Max: " << max << endl;
  cout << "Solution: " << solution.transpose() << endl;
}
