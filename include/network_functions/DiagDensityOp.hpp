#pragma once

#include "Array.hpp"
#include "types.h"

namespace rbm_on_gpu {


struct DiagDensityOp {
    unsigned int    sub_system_size;
    Array<double>   diag_densities_ar;

    inline DiagDensityOp(const unsigned int sub_system_size, const bool gpu)
        :
        sub_system_size(sub_system_size),
        diag_densities_ar(1u << sub_system_size, gpu) {}

    template<typename Psi_t, typename Operator_t, typename SpinEnsemble>
    void operator()(const Psi_t& psi, const Operator_t& op, SpinEnsemble& spin_ensemble);

};


} // namespace rbm_on_gpu
