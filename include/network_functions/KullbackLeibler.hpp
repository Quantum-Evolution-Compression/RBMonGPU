#pragma once

#include "Spins.h"
#include "Array.hpp"
#include "types.h"

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pytensor.hpp"
#endif // __CUDACC__

#include <complex>
// #include <memory>


namespace rbm_on_gpu {

namespace kernel {

class KullbackLeibler {
public:
    bool gpu;

    complex_t*  log_ratio;
    double*     log_ratio_abs2;
    complex_t*  O_k;
    complex_t*  log_ratio_O_k;

    template<bool compute_gradient, typename Psi_t, typename Psi_t_prime, typename SpinEnsemble>
    void compute_averages(
        const Psi_t& psi, const Psi_t_prime& psi_prime, const SpinEnsemble& spin_ensemble
    ) const;
};

} // namespace kernel


class KullbackLeibler : public kernel::KullbackLeibler {
private:
    const unsigned int  num_params;

    Array<complex_t> log_ratio_ar;
    Array<double>    log_ratio_abs2_ar;
    Array<complex_t> O_k_ar;
    Array<complex_t> log_ratio_O_k_ar;

    void clear();

public:
    KullbackLeibler(const unsigned int num_params, const bool gpu);

    template<typename Psi_t, typename Psi_t_prime, typename SpinEnsemble>
    double value(
        const Psi_t& psi, const Psi_t_prime& psi_prime, const SpinEnsemble& spin_ensemble
    );

    template<typename Psi_t, typename Psi_t_prime, typename SpinEnsemble>
    double gradient(
        complex<double>* result, const Psi_t& psi, const Psi_t_prime& psi_prime, const SpinEnsemble& spin_ensemble
    );

#ifdef __PYTHONCC__

    template<typename Psi_t, typename Psi_t_prime, typename SpinEnsemble>
    pair<xt::pytensor<complex<double>, 1u>, double> gradient_py(
        const Psi_t& psi, const Psi_t_prime& psi_prime, const SpinEnsemble& spin_ensemble
    ) {
        xt::pytensor<complex<double>, 1u> grad(std::array<long int, 1u>({(long int)psi_prime.get_num_params()}));

        const double value = this->gradient(grad.data(), psi, psi_prime, spin_ensemble);

        return {grad, value};
    }

#endif // __PYTHONCC__

};

} // namespace rbm_on_gpu
