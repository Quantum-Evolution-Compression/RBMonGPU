#pragma once

#include "types.h"


namespace rbm_on_gpu {

HDINLINE
complex_t my_logcosh(const complex_t z) {
    // return sqrt(0.30102999566 + z*z);
    // return sqrt(1.0 + z*z);
    // return log(2.0 * cosh(z));
    return log(2.0 + 2.0 * z*z);
    // return 2.0 + 2.0 * z*z;
    // return log(1.0 + exp(z));

    // return 0.6932 + 0.5 * z * z;

    // if(abs(z) < 3.0) {
    //     const auto z2 = z * z;

    //     return ((1.0 / 2) * z - (1.0 / 120) * z2) / (1.0 - (1.0 / 10) * z + (1.0 / 120) * z2);
    // }

    // return log((exp(z) - 1.0) / z);
}

HDINLINE
complex_t my_tanh(const complex_t z) {
    // return z / (0.30102999566 + z*z);
    // return z / (1.0 + z*z);
    // return tanh(z);
    return 2.0 * z / (1.0 + z*z);
    // const auto e_z = exp(z);
    // return e_z + (1.0 + e_z);
    // return z;

    // if(abs(z) < 3.0) {
    //     const auto z2 = z * z;

    //     return (1.0 / 2 + (1.0 / 12) * z + (1.0 / 120) * z2) / (1.0 + (1.0 / 60) * z2);
    // }

    // const auto exp_z = exp(z);
    // return -1.0 / z + exp_z / (exp_z - 1.0);
}


} // namespace rbm_on_gpu
