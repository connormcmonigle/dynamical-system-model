#pragma once
#include <tuple>
#include <iostream>
#include <utility>

#include "util.h"

namespace dyn{

template<typename I>
struct weights{
  using info = I;
  typename info::out_mat_t m_out{};
  typename info::out_vec_t b_out{};

  typename info::in_mat_t m_in{};
  typename info::latent_vec_t b_in{};

  typename info::latent_mat_t m_latent{};
  typename info::latent_mat_t m_latent_1{};
  typename info::latent_mat_t m_latent_2{};

  template<typename F>
  void over(F&& f){
    f(m_out);
    f(b_out);

    f(m_in);
    f(b_in);

    f(m_latent);
    f(m_latent_1);
    f(m_latent_2);
  }

  weights(){
    over([](auto& in){ in.setZero(); });
  }

  static weights<I> random(){
    weights<I> result{};
    result.over([](auto& w){ w.setRandom(); });
    return result;
  }

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
    const typename info::out_vec_t out = w.m_out * x + w.b_out;
    return forward_t(out, x + _dx_dt(state) * dt);
  }

  forward_t time_reverse(const backward_t& state) const {
    const auto&[env, x] = state;
    const typename info::out_vec_t out = w.m_out * x + w.b_out;
    return forward_t(out, x - _dx_dt(state) * dt);
  }

  backward_t backward(const backward_t& f_state, const forward_t& b_state){
    const auto&[env, x] = f_state;
    const auto&[env_grad, x_grad] = b_state;

    grad.m_out += env_grad * x.transpose() * dt;
    grad.b_out += env_grad * dt;
    grad.m_in += x_grad * env.transpose() * dt;
    grad.b_in += x_grad * dt;
    grad.m_latent += x_grad * x.transpose() * dt;

    const typename info::latent_vec_t left =  (w.m_latent_1 * x);
    const typename info::latent_vec_t right = (w.m_latent_2 * x);

    grad.m_latent_1 += right.cwiseProduct(x_grad) * x.transpose();
    grad.m_latent_2 +=  left.cwiseProduct(x_grad) * x.transpose();

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

  model(const typename info::real_type& dt_) : dt{dt_} {}

  template<typename ... Args>
  static model<I> random(Args&& ... args){
    model<I> result(std::forward<Args>(args)...);
    result.w = weights<I>::random();
    return result;
  }
};

template<typename I>
std::ostream& operator<<(std::ostream& os, weights<I>& w){
  os << "m_output:\n" << w.m_out << "\n\n";
  os << "b_output:\n" << w.b_out << "\n\n";
  os << "m_input:\n" << w.m_in << "\n\n";
  os << "b_input:\n" << w.b_in << "\n\n";
  os << "m_latent:\n" << w.m_latent << "\n\n";
  os << "m_latent_1:\n" << w.m_latent_1 << "\n\n";
  os << "m_latent_2:\n" << w.m_latent_2 << "\n\n";
  return os;
}

template<typename I>
std::ostream& operator<<(std::ostream& os, model<I>& model){
  os << "model.w:\n" << model.w << "\n\n";
  os << "model.grad:\n" << model.grad << "\n\n";
  return os;
}

}
