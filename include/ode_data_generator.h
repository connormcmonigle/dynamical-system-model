#include <iostream>
#include <random>

#include <Eigen/Dense>

namespace ode{

namespace { using namespace Eigen; }

template<typename Config>
struct train_pair{
  Matrix<typename Config::real_type, Config::dim, 1> input;
  Matrix<typename Config::real_type, Config::dim, 1> output;
};

template<typename Config>
struct trajectory_iterator{
  Config c;
  size_t step{0};
  Matrix<typename Config::real_type, Config::dim, 1> X;

  trajectory_iterator<Config>& operator++(){
    X += c.f(X) * c.dt;
    ++step;
    return *this;
  }

  trajectory_iterator<Config>& operator--(){
    X -= dX_dt(X, c.mu) * c.dt;
    --step;
    return *this;
  }

  train_pair<Config> operator*() const {
    return train_pair<Config>{X, X + c.f(X) * c.dt};
  }

  bool operator==(const trajectory_iterator& other){ return step == other.step; }
  bool operator!=(const trajectory_iterator& other){ return !(*this == other); }

  template<typename T>
  trajectory_iterator(const Config& c_, size_t step_, T&& X_) : c{c_}, step{step_}, X{X_} {}
};

template<typename Config>
struct trajectory{
  Config c;
  Matrix<typename Config::real_type, Config::dim, 1> X;

  trajectory_iterator<Config> begin() const { return trajectory_iterator<Config>(c, 0, X); }
  trajectory_iterator<Config> end() const { return trajectory_iterator<Config>(c, c.steps, X); }

  trajectory(const Config& c_, const Matrix<typename Config::real_type, Config::dim, 1>& X_) : c{c_}, X{X_} {}
};

template<typename Config>
struct data_generator{
  static constexpr int dim = Config::dim;
  static constexpr int input_dim = dim;
  static constexpr int output_dim = dim;
  Config c;

  std::mt19937 generator{std::random_device()()};
  std::uniform_real_distribution<typename Config::real_type> distribution{-0.01, 0.01};

  void grow_domain(){
    constexpr typename Config::real_type growth_factor = 1.5;
    if(distribution.max() < 4.0){
      const typename Config::real_type min = distribution.min() * growth_factor;
      const typename Config::real_type max = distribution.max() * growth_factor;
      distribution = decltype(distribution)(min, max);
    }
  }

  typename Config::real_type dt() const { return c.dt; }

  template<typename T>
  T gradient(const T& true_, const T& pred_) const {
    return (2.0 * (pred_ - true_)).eval() * dt();
  }

  template<typename T>
  typename Config::real_type error(const T& true_, const T& pred_) const {
    return ((pred_ - true_).eval()).squaredNorm();
  }

  trajectory<Config> get_trajectory(){
    Matrix<typename Config::real_type, Config::dim, 1> X{};
    X = X.unaryExpr([this](auto){ return distribution(generator); });
    return trajectory<Config>(c, X);
  }

  data_generator(Config c_) : c{c_} {}
};

}