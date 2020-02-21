#include <iostream>
#include <util.h>
#include <model.h>


int main(){
  using sys = dyn::model<util::info<long double, 4, 10, 16>>;
  auto m = sys::random(0.01);
  std::cout << m << std::endl;
  auto[output, latent] = m.forward(sys::backward_t());
  auto[in_grad, latent_grad] = m.backward(sys::backward_t(), sys::forward_t());
  std::cout << "output: " << std::endl << output << std::endl;
  std::cout << "latent: " << std::endl << latent << std::endl;
  std::cout << "in_grad: " << std::endl << in_grad << std::endl;
  std::cout << "latent_grad: " << std::endl << latent_grad << std::endl;
}
