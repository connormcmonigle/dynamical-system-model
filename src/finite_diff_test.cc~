#include <ctime>
#include <cstdlib>
#include <Eigen/Dense>

#include <util.h>
#include <model.h>

using namespace Eigen;

template<typename F, typename I>
MatrixXd finite_diff_jacobian(F&& f, I& at){
  constexpr double dx = 0.001;
  const auto f_ref = f(at);
  MatrixXd result(f_ref.rows(), at.rows()); result.setZero();
  std::cout << result.cols() << " -> " << result.rows() << std::endl;
  for(int i(0); i < result.cols(); ++i){
    at(i) += dx;
    result.col(i) = (f(at) - f_ref) / dx;
    at(i) -= dx;
  }
  return result;
}

template<typename M>
void test_latent_to_latent_gradient(M model){
  std::cout << "test latent -> latent gradient: " << std::endl;

  typename M::info::in_vec_t input{}; input.setRandom();
  typename M::info::latent_vec_t latent{}; latent.setRandom();

  typename M::info::latent_vec_t latent_grad{}; latent_grad.setRandom();
  typename M::info::out_vec_t out_grad{}; out_grad.setZero();

  auto f = [input, latent, &model](const auto& x){
    auto[output, latent_next] = model.forward(typename M::backward_t(input, x));
    return latent_next;
  };

  auto[in_impl, latent_impl] = model.backward(
    typename M::backward_t(input, latent),
    typename M::forward_t(out_grad, latent_grad)
  );

  std::cout << "\ncorrect:\n" << 
  (latent_grad.transpose() * finite_diff_jacobian(f, latent)).transpose() <<
  std::endl << std::endl;

  std::cout << "\nimplementation:\n" <<
  latent_impl <<
  std::endl << std::endl;
  
}

template<typename M>
void test_latent_to_output_gradient(M model){
  std::cout << "test latent -> output gradient: " << std::endl;

  typename M::info::in_vec_t input{}; input.setRandom();
  typename M::info::latent_vec_t latent{}; latent.setRandom();

  typename M::info::latent_vec_t latent_grad{}; latent_grad.setZero();
  typename M::info::out_vec_t out_grad{}; out_grad.setRandom();    

  auto f = [input, latent, &model](const auto& x){
    auto[output, latent_next] = model.forward(typename M::backward_t(input, x));
    return output;
  };

  auto[in_impl, latent_impl] = model.backward(
    typename M::backward_t(input, latent),
    typename M::forward_t(out_grad, latent_grad)
  );

  std::cout << "\ncorrect:\n" << 
  (out_grad.transpose() * finite_diff_jacobian(f, latent)).transpose() <<
  std::endl << std::endl;

  std::cout << "\nimplementation:\n" <<
  latent_impl <<
  std::endl << std::endl;
}

template<typename M>
void test_input_to_latent_gradient(M model){
  std::cout << "test input -> latent gradient: " << std::endl;
  typename M::info::in_vec_t input{}; input.setRandom();
  typename M::info::latent_vec_t latent{}; latent.setRandom();

  typename M::info::latent_vec_t latent_grad{}; latent_grad.setRandom();
  typename M::info::out_vec_t out_grad{}; out_grad.setZero(); 
  

  auto f = [input, latent, &model](const auto& x){
    auto[output, latent_next] = model.forward(typename M::backward_t(x, latent));
    return latent_next;
  };

  auto[in_impl, latent_impl] = model.backward(
    typename M::backward_t(input, latent),
    typename M::forward_t(out_grad, latent_grad)
  );

  std::cout << "\ncorrect:\n" << 
  (latent_grad.transpose() * finite_diff_jacobian(f, input)).transpose() <<
  std::endl << std::endl;

  std::cout << "\nimplementation:\n" <<
  in_impl <<
  std::endl << std::endl;

}

int main(){
  std::srand (std::time(nullptr));
  auto dm = dyn::model<util::info<double, 4, 5, 6>>::random(0.25);
  test_latent_to_latent_gradient(dm);
  test_latent_to_output_gradient(dm);
  test_input_to_latent_gradient(dm);
}
