#pragma once

#include "types.h"


namespace rbm_on_gpu {

HDINLINE
complex_t my_logcosh(const complex_t z) {
    // return sqrt(0.30102999566 + z*z);

    // return z*z * (1.0 / 2.0) - z*z*z*z * (1.0 / 12.0);
    return z;

    // return sqrt(1.0 + z*z) - 1.0;

    // return complex_t(0.0, -1.0) * log(cosh(z)) + complex_t(0.0, -0.346574);
    // return log(cosh(z));
    // return tanh(z);

    // seems to be dangerous. Does not work for a SW-generator applied on an initial state.
    // return log(1.0 + z*z);

    // return 2.0 + 2.0 * z*z;
    // return log(1.0 + exp(z));

    // return 0.6932 + 0.5 * z * z;

    // if(abs(z) < 3.0) {
    //     const auto z2 = z * z;

    //     return ((1.0 / 2) * z - (1.0 / 120) * z2) / (1.0 - (1.0 / 10) * z + (1.0 / 120) * z2);
    // }

    // return log((exp(z) - 1.0) / z);

    // const auto sign = z.real() > 0.0 ? 1.0 : -1.0;

    // return sign * z + (1.81168 - sign * 1.22741 * z) / (2.61371 + z * (sign * 2.0 + z)) - 0.693147;

    // return sign * 0.9003320053750442 * z + (
    //     5.49914721954 - sign * 2.16564366435 * z
    // ) / (
    //     9.19376335670885 + z * (sign * 10.2180213465 + z * (7.771429504240965 + z * (sign * 3.746646023906276 + z)))
    // ) - 0.598139;
}

HDINLINE
complex_t my_tanh(const complex_t z) {
    // return z / (0.30102999566 + z*z);
    // return z / (1.0 + z*z);
    // return complex_t(0.0, -1.0) * tanh(z);
    // return complex_t(2.0, 0.0) / (cosh(2.0 * z) + 1.0);
    // return 2.0 * z / (1.0 + z*z);

    // return z - z*z*z * (1.0 / 3.0);
    return complex_t(1.0, 0.0);

    // return z / sqrt(1.0 + z*z);

    // return tanh(z);

    // const auto e_z = exp(z);
    // return e_z + (1.0 + e_z);
    // return z;

    // if(abs(z) < 3.0) {
    //     const auto z2 = z * z;

    //     return (1.0 / 2 + (1.0 / 12) * z + (1.0 / 120) * z2) / (1.0 + (1.0 / 60) * z2);
    // }

    // const auto exp_z = exp(z);
    // return -1.0 / z + exp_z / (exp_z - 1.0);

    // const auto sign = z.real() > 0.0 ? 1.0 : -1.0;
    // const auto denominator = 2.61371 + z * (sign * 2.0 + z);
    // return (
    //     z * (6.83146 + z * (sign * 10.4548 + z * (4.0 + sign * z)))
    // ) / (denominator * denominator);

    // const auto denominator = 9.19376335670885 + z * (sign * 10.218021346543315 + z * (7.771429504240965 + z * (sign * 3.746646023906276 + z)));
    // return (
    //     z * (
    //         83.68563506532087 + z * (
    //             sign * 177.6769746361748 + z * (
    //                 199.24474920889975 + z * (
    //                     sign * 146.36284300074402 + z * (
    //                         70.82878897882324 + z * (
    //                             sign * 26.632014683761202 + z * (
    //                                 6.746450656267947 + sign * 0.9003320053750442 * z
    //     )))))))
    // ) / (denominator * denominator);
}


} // namespace rbm_on_gpu
