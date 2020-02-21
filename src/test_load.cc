#include <iostream>
#include <fstream>
#include <van_der_pol.h>
#include <model.h>
#include <train.h>

int main(){
    auto model = dyn::model<util::info<double, 2, 2, 16>>(0.0);
    std::ifstream save_file("check_pt/model_save.txt");
    save_file >> model;
    std::cout << std::endl << model << std::endl;
}
