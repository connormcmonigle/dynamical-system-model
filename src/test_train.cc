#include <iostream>
#include <fstream>
#include <van_der_pol.h>
#include <model.h>
#include <train.h>

int main(){
    auto data = ode::data_generator(van_der_pol::config{1.5, 0.01, 5000ull});
    auto model = dyn::model<util::info<double, 2, 2, 6>>::random(0.0);
    auto trainer = train::trainer(model, data).set_lr(0.01);
    constexpr int sample_rate = 100;
    constexpr int save_rate = 6000;
    double sum{0.0};
    for(size_t i{0};;++i){
      sum += trainer.update_model();

      if(i !=0 && i % sample_rate == 0){
        std::cout << "\r" << std::flush << sum / static_cast<decltype(sum)>(sample_rate);
        sum = 0.0;
      }

      if(i !=0 && i % save_rate == 0){
        std::fstream save_file("check_pt/model_save_" + std::to_string(i) + ".txt", std::ios::app);
        save_file << trainer.model;
      }

    }
}
