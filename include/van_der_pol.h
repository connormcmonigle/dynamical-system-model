#include <iostream>
#include <random>

#include <Eigen/Dense>

#include <ode_data_generator.h>

namespace van_der_pol{

namespace { using namespace Eigen; }

struct config{
  using real_type = double;
  static constexpr int dim = 2;

  real_type mu{0.5};
  real_type dt{0.001};
  size_t steps{1000};

  template<typename T>
  T f(const T& X) const {
    T result{};
    result << X(1), (mu * (1.0 - X(0)*X(0)) * X(1) - X(0));
    return result;
  }
};

}
