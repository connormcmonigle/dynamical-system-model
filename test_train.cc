#include "van_der_pol.h"
#include "model.h"
#include "train.h"

int main(){
    auto data = van_der_pol::data_generator(van_der_pol::config{0.2, 0.01, 1000ull});
    auto model = dyn::model<util::info<double, 2, 2, 2>>::random(0.0);
    auto trainer = train::trainer(model, data).set_lr(1e-6);
    for(int i(0); i < 10000; ++i){
      trainer.update_model();
      std::cout << trainer.model << std::endl;
    }
}
