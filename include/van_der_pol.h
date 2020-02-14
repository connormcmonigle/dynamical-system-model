#include <iostream>
#include <random>

#include "Eigen/Dense"

namespace van_der_pol{

namespace { using namespace Eigen; }

struct config{
    using real_type = double;
    static constexpr int input_dim = 2;
    static constexpr int output_dim = 2;
    real_type mu{0.0};
    real_type dt{0.001};
    size_t steps{1000ull};
};

template<typename T>
T dX_dt(const T& X, const config::real_type& mu){
    T result{};
    result << X(1), (mu * (1.0 - X(0)*X(0)) * X(1) - X(0));
    return result;
}

struct train_pair{
    Matrix<config::real_type, config::output_dim, 1> input;
    Matrix<config::real_type, config::output_dim, 1> output;
};

struct trajectory_iterator{
    config c;
    size_t step{0ull};
    Matrix<config::real_type, config::output_dim, 1> X;

    trajectory_iterator& operator++(){
        X += dX_dt(X, c.mu) * c.dt;
        ++step;
        return *this;
    }

    trajectory_iterator& operator--(){
        X -= dX_dt(X, c.mu) * c.dt;
        --step;
        return *this;
    }

    train_pair operator*() const {
        return train_pair{X, X + dX_dt(X, c.mu) * c.dt};
    }

    bool operator==(const trajectory_iterator& other){ return step == other.step; }
    bool operator!=(const trajectory_iterator& other){ return !(*this == other); }

    template<typename T>
    trajectory_iterator(config c_, size_t step_, T&& X_) : c{c_}, step{step_}, X{X_} {}
};

struct trajectory{
    config c;
    Matrix<config::real_type, config::output_dim, 1> X;

    trajectory_iterator begin() const { return trajectory_iterator(c, 0, X); }
    trajectory_iterator end() const { return trajectory_iterator(c, c.steps, X); }

    trajectory(const config& c_, config::real_type x, config::real_type v) : c{c_} { X << x, v; }
};

struct data_generator{
    static constexpr int input_dim = config::input_dim;
    static constexpr int output_dim = config::output_dim;
    config c;

    std::mt19937 generator{std::random_device()()};
    std::uniform_real_distribution<config::real_type> distribution{-5.0, 5.0};

    config::real_type dt() const { return c.dt; }

    template<typename T>
    T gradient(const T& true_, const T& pred_) const {
        return (2.0 * (pred_ - true_)).eval() * dt();
    }

    template<typename T>
    config::real_type error(const T& true_, const T& pred_) const {
      return ((pred_ - true_).eval()).squaredNorm();
    }

    trajectory get_trajectory(){
        const auto x = distribution(generator);
        const auto v = distribution(generator);
        return trajectory(c, x, v);
    }

    data_generator(config c_) : c{c_} {}

};

}
