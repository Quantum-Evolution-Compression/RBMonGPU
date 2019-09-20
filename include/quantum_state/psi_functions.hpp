#pragma once

#include "types.h"


namespace rbm_on_gpu {

HDINLINE
complex_t my_logcosh(const complex_t z) {
    // return sqrt(0.30102999566 + z*z);
    return sqrt(1.0f + z*z) - 1.0f;
    // return log(2.0f * cosh(z));
    // return log(2.0f + 2.0f * z*z);
    // return 2.0f + 2.0f * z*z;
    // return log(1.0f + exp(z));

    // return 0.6932 + 0.5 * z * z;

    // if(abs(z) < 3.0f) {
    //     const auto z2 = z * z;

    //     return ((1.0f / 2) * z - (1.0f / 120) * z2) / (1.0f - (1.0f / 10) * z + (1.0f / 120) * z2);
    // }

    // return log((exp(z) - 1.0f) / z);
}

HDINLINE
complex_t my_tanh(const complex_t z) {
    // return z / (0.30102999566 + z*z);
    return z / (1.0f + z*z);
    // return tanh(z);
    // return 2.0f * z / (1.0f + z*z);
    // const auto e_z = exp(z);
    // return e_z + (1.0f + e_z);
    // return z;

    // if(abs(z) < 3.0f) {
    //     const auto z2 = z * z;

    //     return (1.0f / 2 + (1.0f / 12) * z + (1.0f / 120) * z2) / (1.0f + (1.0f / 60) * z2);
    // }

    // const auto exp_z = exp(z);
    // return -1.0f / z + exp_z / (exp_z - 1.0f);
}


} // namespace rbm_on_gpu
