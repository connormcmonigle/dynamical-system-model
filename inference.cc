#include <iostream>
#include <fstream>

#include "util.h"
#include "model.h"


int main(){
    std::fstream load_file("check_pt/model_save_48000.txt");
    using M = dyn::model<util::info<double, 2, 2, 16>>;
    M m{0.01}; load_file >> m;
    std::cout << m << std::endl;
    M::info::latent_vec_t latent{}; latent.setZero();
    M::info::in_vec_t input{}; input(0) = input(1) = 0.5;
    for(size_t i(0); i < 2000; ++i){
        const auto [next_input, next_latent] = m.forward(M::backward_t(input, latent));
        input = next_input;
        latent = next_latent;
        std::cout << input(1) << std::endl;
    }
}