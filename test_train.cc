#include "van_der_pol.h"
#include "model.h"
#include "train.h"

int main(){
    auto data = van_der_pol::data_generator(van_der_pol::config{0.2, 0.01, 1000ull});
    auto model = dyn::model<util::info<double, 2, 2, 2>>::random(0.0);
    auto trainer = train::trainer(model, data).set_lr(0.02);
    for(size_t i{0};;++i){
      if(i % 1000 == 0) std::cout << trainer.update_model() << std::endl;
      //std::cout << trainer.model.w << std::endl;
    }
}
