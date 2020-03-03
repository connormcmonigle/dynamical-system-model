#include <iostream>
#include <random>

#include <Eigen/Dense>

namespace lorenz{

namespace { using namespace Eigen; }

struct config{
  using real_type = double;
  static constexpr int dim = 3;

  real_type sigma{10.0};
  real_type rho{28.0};
  real_type beta{8.0/3.0};

  real_type dt{0.001};
  size_t steps{200};

  template<typename T>
  T f(const T& X) const {
    T result{};
    result <<
      sigma * (X(1) - X(0)),
      X(0) * (rho - X(2)) - X(1),
      X(0) * X(1) - beta * X(2);
    return result;
  }
};

}