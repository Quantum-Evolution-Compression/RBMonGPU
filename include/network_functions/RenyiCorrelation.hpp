#pragma once

#include "operator/Operator.hpp"
#include "Array.hpp"


namespace rbm_on_gpu {


struct RenyiCorrelation {
    bool        gpu;
    Array<double> result_ar;

    inline RenyiCorrelation(const bool gpu) : result_ar(1, gpu) {}

    template<typename Psi_t, typename SpinEnsemble>
    double operator()(const Psi_t& psi, const Operator& U_A, SpinEnsemble& spin_ensemble);
};


} // namespace rbm_on_gpu
