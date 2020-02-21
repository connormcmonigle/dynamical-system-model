#include <iostream>
#include <fstream>

#include <util.h>
#include <model.h>


int main(){
    std::string file_name; std::cout << "file_name :: "; std::cin >> file_name;
    std::fstream load_file(file_name);
    using M = dyn::model<util::info<double, 2, 2, 6>>;
    M m{0.01}; load_file >> m;
    std::cout << m << std::endl;
    M::info::latent_vec_t latent{}; latent.setZero();
    M::info::in_vec_t input{}; input(0) = input(1) = 0.5;
    std::fstream csv_output("output.csv", std::ios::app);
    csv_output << "x, y" << std::endl;
    for(size_t i(0); i < 10000; ++i){
        const auto [next_input, next_latent] = m.forward(M::backward_t(input, latent));
        input = next_input;
        latent = next_latent;
        csv_output << input(0) << ", " << input(1) << std::endl;
    }
}
