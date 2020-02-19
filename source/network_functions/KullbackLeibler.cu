#include "network_functions/KullbackLeibler.hpp"
#include "spin_ensembles/ExactSummation.hpp"
#include "spin_ensembles/MonteCarloLoop.hpp"
#include "quantum_state/PsiDeep.hpp"
#include "quantum_state/PsiClassical.hpp"

#include <cstring>
#include <math.h>


namespace rbm_on_gpu {

namespace kernel {


template<bool compute_gradient, typename Psi_t, typename Psi_t_prime, typename SpinEnsemble>
void kernel::KullbackLeibler::compute_averages(
    const Psi_t& psi, const Psi_t_prime& psi_prime, const SpinEnsemble& spin_ensemble
) const {
    const auto num_params = psi_prime.get_num_params();

    const auto this_ = *this;
    const auto psi_kernel = psi.get_kernel();
    const auto psi_prime_kernel = psi_prime.get_kernel();

    spin_ensemble.foreach(
        psi,
        [=] __device__ __host__ (
            const unsigned int spin_index,
            const Spins spins,
            const complex_t log_psi,
            typename Psi_t::Angles& angles,
            const double weight
        ) {
            #include "cuda_kernel_defines.h"

            SHARED typename Psi_t_prime::Angles angles_prime;
            angles_prime.init(psi_prime_kernel, spins);

            SHARED complex_t log_psi_prime;
            psi_prime_kernel.log_psi_s(log_psi_prime, spins, angles_prime);

            SINGLE
            {
                generic_atomicAdd(this_.log_ratio, weight * (log_psi_prime - log_psi));
                generic_atomicAdd(this_.log_ratio_abs2, weight * abs2(log_psi_prime - log_psi));
            }

            if(compute_gradient) {
                psi_prime_kernel.foreach_O_k(
                    spins,
                    angles_prime,
                    [&](const unsigned int k, const complex_t& O_k_element) {
                        generic_atomicAdd(
                            &this_.O_k[k],
                            weight * conj(O_k_element)
                        );
                        generic_atomicAdd(
                            &this_.log_ratio_O_k[k],
                            weight * (log_psi_prime - log_psi) * conj(O_k_element)
                        );
                    }
                );
            }
        },
        max(psi.get_width(), psi_prime.get_width())
    );
}

} // namespace kernel

KullbackLeibler::KullbackLeibler(const unsigned int num_params, const bool gpu)
      : num_params(num_params),
        log_ratio_ar(1, gpu),
        log_ratio_abs2_ar(1, gpu),
        O_k_ar(num_params, gpu),
        log_ratio_O_k_ar(num_params, gpu)
    {
    this->gpu = gpu;

    this->log_ratio = this->log_ratio_ar.data();
    this->log_ratio_abs2 = this->log_ratio_abs2_ar.data();
    this->O_k = this->O_k_ar.data();
    this->log_ratio_O_k = this->log_ratio_O_k_ar.data();
}


void KullbackLeibler::clear() {
    this->log_ratio_ar.clear();
    this->log_ratio_abs2_ar.clear();
    this->O_k_ar.clear();
    this->log_ratio_O_k_ar.clear();
}


template<typename Psi_t, typename Psi_t_prime, typename SpinEnsemble>
double KullbackLeibler::value(
    const Psi_t& psi, const Psi_t_prime& psi_prime, const SpinEnsemble& spin_ensemble
) {
    this->clear();
    this->compute_averages<false>(psi, psi_prime, spin_ensemble);

    this->log_ratio_ar.update_host();
    this->log_ratio_abs2_ar.update_host();

    this->log_ratio_ar.front() /= spin_ensemble.get_num_steps();
    this->log_ratio_abs2_ar.front() /= spin_ensemble.get_num_steps();

    return sqrt(this->log_ratio_abs2_ar.front() - abs2(this->log_ratio_ar.front()));
}


template<typename Psi_t, typename Psi_t_prime, typename SpinEnsemble>
double KullbackLeibler::gradient(
    complex<double>* result, const Psi_t& psi, const Psi_t_prime& psi_prime, const SpinEnsemble& spin_ensemble
) {
    this->clear();
    this->compute_averages<true>(psi, psi_prime, spin_ensemble);

    this->log_ratio_ar.update_host();
    this->log_ratio_abs2_ar.update_host();
    this->O_k_ar.update_host();
    this->log_ratio_O_k_ar.update_host();

    this->log_ratio_ar.front() /= spin_ensemble.get_num_steps();
    this->log_ratio_abs2_ar.front() /= spin_ensemble.get_num_steps();

    const auto value = sqrt(this->log_ratio_abs2_ar.front() - abs2(this->log_ratio_ar.front()));

    for(auto k = 0u; k < this->num_params; k++) {
        this->O_k_ar.at(k) *= 1.0 / spin_ensemble.get_num_steps();
        this->log_ratio_O_k_ar.at(k) *= 1.0 / spin_ensemble.get_num_steps();

        result[k] = (
            this->log_ratio_O_k_ar[k] - this->log_ratio_ar.front() * this->O_k_ar[k]
        ).to_std() / value;
    }

    return value;
}


template double KullbackLeibler::value(
    const PsiDeep& psi, const PsiDeep& psi_prime,
    const ExactSummation& spin_ensemble
);
template double KullbackLeibler::value(
    const PsiDeep& psi, const PsiDeep& psi_prime,
    const MonteCarloLoop& spin_ensemble
);

template double KullbackLeibler::gradient(
    complex<double>* result, const PsiDeep& psi, const PsiDeep& psi_prime, const ExactSummation& spin_ensemble
);
template double KullbackLeibler::gradient(
    complex<double>* result, const PsiDeep& psi, const PsiDeep& psi_prime, const MonteCarloLoop& spin_ensemble
);


template double KullbackLeibler::value(
    const PsiClassical& psi, const PsiDeep& psi_prime, const ExactSummation& spin_ensemble
);
template double KullbackLeibler::value(
    const PsiClassical& psi, const PsiDeep& psi_prime, const MonteCarloLoop& spin_ensemble
);

template double KullbackLeibler::gradient(
    complex<double>* result, const PsiClassical& psi, const PsiDeep& psi_prime, const ExactSummation& spin_ensemble
);
template double KullbackLeibler::gradient(
    complex<double>* result, const PsiClassical& psi, const PsiDeep& psi_prime, const MonteCarloLoop& spin_ensemble
);

} // namespace rbm_on_gpu
