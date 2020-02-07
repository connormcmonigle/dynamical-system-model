#include <iostream>
#include <fstream>
#include "van_der_pol.h"
#include "model.h"
#include "train.h"

int main(){
    auto data = van_der_pol::data_generator(van_der_pol::config{0.2, 0.01, 1000ull});
    auto model = dyn::model<util::info<double, 2, 2, 16>>::random(0.0);
    auto trainer = train::trainer(model, data).set_lr(0.02);
    for(size_t i{0};;++i){
      trainer.update_model();
      if(i % 1000 == 0){
        std::cout << trainer.update_model() << std::endl;
        std::fstream save_file("check_pt/model_save_" + std::to_string(i) + ".txt", std::ios::app);
        save_file << trainer.model;
      }
    }
}
