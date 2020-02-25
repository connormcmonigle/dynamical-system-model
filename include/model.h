#pragma once
#include <tuple>
#include <iostream>
#include <utility>
#include <string_view>
#include <string>
#include <sstream>
#include <cassert>

#include <util.h>

namespace dyn{

template<typename F, typename ... Args>
void over_weights(F&& f, Args&& ... args){
  f((args.m_out)...);
  f((args.b_out)...);
  f((args.m_through)...);

  f((args.m_in)...);
  f((args.b_in)...);

  f((args.m_latent)...);
  f((args.m_latent_1)...);
  f((args.m_latent_2)...);
}

template<typename I>
struct weights{
  using info = I;
  typename info::out_mat_t m_out{};
  typename info::out_vec_t b_out{};
  typename info::through_mat_t m_through{};

  typename info::in_mat_t m_in{};
  typename info::latent_vec_t b_in{};

  typename info::latent_mat_t m_latent{};
  typename info::latent_mat_t m_latent_1{};
  typename info::latent_mat_t m_latent_2{};

  template<typename F>
  void over(F&& f){
    return over_weights(std::forward<F>(f), *this);
  }

  weights(){
    over([](auto& in){ in.setZero(); });
  }

  static weights<I> random(){
    const typename info::real_type init_factor = 0.02;
    weights<I> result{};
    result.over([init_factor](auto& w){
      w.setRandom();
      w *= init_factor;
    });
    return result;
  }

};

struct weight_names{
  std::string_view m_out{"m_out"};
  std::string_view b_out{"b_out"};
  std::string_view m_through{"m_through"};

  std::string_view m_in{"m_in"};
  std::string_view b_in{"b_in"};

  std::string_view m_latent{"m_latent"};
  std::string_view m_latent_1{"m_latent_1"};
  std::string_view m_latent_2{"m_latent_2"};
};

template<typename I>
struct model{
  using info = I;
  using forward_t = std::tuple<typename info::out_vec_t, typename info::latent_vec_t>;
  using backward_t = std::tuple<typename info::in_vec_t, typename info::latent_vec_t>;  


  weights<info> w{};
  weights<info> grad{};
  typename info::real_type dt;

  void set_dt(const typename info::real_type& _dt){ dt = _dt; }


  typename info::latent_vec_t _dx_dt(const backward_t& state) const {
    const auto&[env, x] = state;
    const typename info::latent_vec_t in = w.m_in * env + w.b_in;
    const typename info::latent_vec_t latent = w.m_latent * x;
    const typename info::latent_vec_t latent_2nd = (w.m_latent_1 * x).cwiseProduct(w.m_latent_2 * x);
    return in + latent + latent_2nd;
  }

  forward_t forward(const backward_t& state) const {
    const auto&[env, x] = state;
    const typename info::out_vec_t out = w.m_through * env + w.m_out * x + w.b_out;
    return forward_t(out, x + _dx_dt(state) * dt);
  }

  forward_t time_reverse(const backward_t& state) const {
    const auto&[env, x] = state;
    const typename info::out_vec_t out = w.m_through * env+  w.m_out * x + w.b_out;
    return forward_t(out, x - _dx_dt(state) * dt);
  }

  backward_t backward(const backward_t& f_state, const forward_t& b_state){
    const auto&[env, x] = f_state;
    const auto&[env_grad, x_grad] = b_state;

    grad.m_out += env_grad * x.transpose() * dt;
    grad.b_out += env_grad * dt;
    grad.m_through += env_grad * env.transpose() * dt;
    grad.m_in += x_grad * env.transpose() * dt;
    grad.b_in += x_grad * dt;
    grad.m_latent += x_grad * x.transpose() * dt;

    const typename info::latent_vec_t left =  (w.m_latent_1 * x);
    const typename info::latent_vec_t right = (w.m_latent_2 * x);

    grad.m_latent_1 += right.cwiseProduct(x_grad) * x.transpose() * dt;
    grad.m_latent_2 +=  left.cwiseProduct(x_grad) * x.transpose() * dt;

    const typename info::latent_mat_t hadamard_jacobian =
      (left.template replicate<1, info::latent_dim>()).cwiseProduct(w.m_latent_2) + 
      (right.template replicate<1, info::latent_dim>()).cwiseProduct(w.m_latent_1);

    const typename info::latent_vec_t x_grad_next =
      x_grad + 
      (env_grad.transpose() * w.m_out).transpose() +
      (x_grad.transpose() * w.m_latent).transpose() * dt +
      (x_grad.transpose() * hadamard_jacobian).transpose() * dt;

    const typename info::in_vec_t in_grad_next =
      (x_grad.transpose() * w.m_in).transpose() * dt;

    return backward_t(in_grad_next, x_grad_next);
  }

  void clear_grad(){
    grad.over([](auto& in){ in.setZero(); });
  }

  void step_grad(typename info::real_type learning_rate){
    auto update_rule = [learning_rate](auto& w_, const auto& grad_){
      w_ -= learning_rate * grad_;
    };
    over_weights(update_rule, w, grad);
  }

  model(const typename info::real_type& dt_) : dt{dt_} {}

  template<typename ... Args>
  static model<I> random(Args&& ... args){
    model<I> result(std::forward<Args>(args)...);
    result.w = weights<I>::random();
    return result;
  }
};

template<typename I>
std::ostream& operator<<(std::ostream& os, const weights<I>& w){
  over_weights([&os](const std::string_view& name, const auto& weight){
    os << name << '\n' << weight << "\n\n";
  }, weight_names{}, w);
  return os;
}

template<typename I>
std::istream& operator>>(std::istream& is, weights<I>& w){
  auto load_weight = [&is](const std::string_view& name, auto& weight){
    std::string line{}; std::getline(is, line);
    std::cout << "loading: " << line << ", " << name << std::endl;
    assert((line == name));
    for(Eigen::Index row{0}; std::getline(is, line) && row < weight.rows(); ++row){
      std::istringstream ss(line);
      std::string val{};
      for(Eigen::Index col{0}; col < weight.cols(); ++col){
        ss >> weight(row, col);
      }
    }
  };
  over_weights(load_weight, weight_names{}, w);
  return is;
}

template<typename I>
std::ostream& operator<<(std::ostream& os, model<I>& model){
  os << model.w;
  return os;
}

template<typename I>
std::istream& operator>>(std::istream& is, model<I>& model){
  is >> model.w;
  return is;
}

}
