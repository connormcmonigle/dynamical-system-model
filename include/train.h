#pragma once
#include <iostream>
#include <utility>
#include <vector>


namespace train{

template<typename I>
struct sample{
  using info = I;
  typename info::real_type t;
  typename info::in_vec_t input;
  typename info::out_vec_t gradient;
  typename info::latent_vec_t latent;
};

template<typename M, typename D>
struct trainer{
  using info = typename M::info;
  typename info::real_type learning_rate{0.0};
  M model;
  D data;

  std::vector<sample<info>> history{};


  template<typename T>
  trainer<M, D>& set_lr(T&& lr){
    learning_rate = lr;
    return *this;
  }

  typename info::real_type update_model(){
    typename info::latent_vec_t latent; latent.setZero();
    auto trajectory = data.get_trajectory();
    typename info::real_type t{0.0};
    typename info::real_type err{0.0};
    for(auto&&[input, exp_out] : trajectory){
      const auto[out, next_latent] = model.forward(typename M::backward_t(input, latent));
      const auto gradient = data.gradient(exp_out, out);
      const auto error = data.error(exp_out, out);
      history.push_back(sample<info>{t, input, gradient, latent});
      latent = next_latent;
      err += error;
      t += data.dt();
    }
    
    typename info::latent_vec_t latent_grad; latent_grad.setZero();
    for(auto iter = history.rbegin(); iter != history.rend(); ++iter){
      //std::cout << iter -> gradient << std::endl;
      const auto grad_info = typename M::forward_t(iter -> gradient, latent_grad);
      const auto state_info = typename M::backward_t(iter -> input, iter -> latent);
      // ignore gradient of loss w.r.t input for now.
      const auto[_, latent_grad_next] = model.backward(state_info, grad_info);
      latent_grad = latent_grad_next;
    }
    model.step_grad(learning_rate);
    model.clear_grad();
    history.clear();
    return err;
  }

  trainer(M m, D d) : model(m), data(d) {
    static_assert(info::output_dim == D::output_dim);
    static_assert(info::input_dim == D::input_dim);
    model.set_dt(data.dt());
  }
};

}


