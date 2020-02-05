#include "van_der_pol.h"
#include "model.h"
#include "train.h"

int main(){
    auto data = van_der_pol::data_generator(van_der_pol::config{0.5, 0.1, 100ull});
    auto model = dyn::model<util::info<double, 2, 2, 2>>(0.0);
    auto trainer = train::trainer(model, data).set_lr(0.1);
    trainer.update_model();
}
