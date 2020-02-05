#include "van_der_pol.h"
#include "model.h"
#include "train.h"

int main(){
    auto data = van_der_pol::data_generator(van_der_pol::config{1.0, 0.0001, 100ull});
    auto model = dyn::model<util::info<double, 2, 2, 2>>::random(0.0);
    auto trainer = train::trainer<decltype(model), decltype(data)>(model, data);
    trainer.update_model();
}