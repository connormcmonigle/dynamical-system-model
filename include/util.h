#pragma once

#include "Eigen/Dense"

namespace util{

namespace { using namespace Eigen; }

template<typename T, int Input, int Output, int Latent>
struct info{
  static constexpr int input_dim = Input;
  static constexpr int output_dim = Output;
  static constexpr int latent_dim = Latent;
  using real_type = T;
  using in_vec_t = Matrix<T, Input, 1>;
  using out_vec_t = Matrix<T, Output, 1>;
  using latent_vec_t = Matrix<T, Latent, 1>;
  using in_mat_t = Matrix<T, Latent, Input>;
  using out_mat_t = Matrix<T, Output, Latent>;
  using latent_mat_t = Matrix<T, Latent, Latent>;
  using through_mat_t = Matrix<T, Output, Input>;
};

}
