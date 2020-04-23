#pragma once

#include "operator/Operator.hpp"
#include "Array.hpp"

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pytensor.hpp"
#endif


namespace rbm_on_gpu {


struct DiagDensityOp {
    unsigned int    sub_system_size;
    Array<double>   diag_densities_ar;

    inline DiagDensityOp(const unsigned int sub_system_size, const bool gpu)
        :
        sub_system_size(sub_system_size),
        diag_densities_ar(1u << sub_system_size, gpu) {}

    template<typename Psi_t, typename SpinEnsemble>
    void operator()(const Psi_t& psi, const Operator& op, SpinEnsemble& spin_ensemble);

#ifdef __PYTHONCC__

    template<typename Psi_t, typename SpinEnsemble>
    inline xt::pytensor<double, 1u> __call__(const Psi_t& psi, const Operator& op, SpinEnsemble& spin_ensemble) {
        (*this)(psi, op, spin_ensemble);
        return this->diag_densities_ar.to_pytensor_1d();
    }

#endif
};


} // namespace rbm_on_gpu
