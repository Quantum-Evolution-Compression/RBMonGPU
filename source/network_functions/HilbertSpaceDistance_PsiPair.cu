#include "network_functions/HilbertSpaceDistance.hpp"
#include "spin_ensembles.hpp"
#include "quantum_states.hpp"

#include <cstring>
#include <math.h>


namespace rbm_on_gpu {

namespace kernel {


template<bool compute_gradient, bool real_gradient, typename Psi_t, typename SpinEnsemble>
void kernel::HilbertSpaceDistance::compute_averages2(
    const Psi_t& psi, const PsiPair& psi_prime, const Operator& operator_,
    const bool is_unitary, const SpinEnsemble& spin_ensemble
) const {
    const auto num_params = psi_prime.get_num_params();

    const auto this_ = *this;
    const auto psi_kernel = psi.get_kernel();
    const auto psi_prime_kernel = psi_prime.get_kernel();
    const auto N = psi.get_num_spins();

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

            SHARED complex_t local_energy;
            operator_.local_energy(local_energy, psi_kernel, spins, log_psi, angles);

            SHARED typename PsiPair::Angles angles_prime;
            angles_prime.init(psi_prime_kernel, spins);

            SHARED complex_t log_psi_prime;
            psi_prime_kernel.log_psi_s(log_psi_prime, spins, angles_prime);

            SHARED complex_t   omega;
            SHARED double      probability_ratio;

            SINGLE
            {
                if(is_unitary) {
                    omega = exp(conj(log_psi_prime - log_psi)) * local_energy;
                    generic_atomicAdd(
                        this_.next_state_norm_avg,
                        weight * (local_energy * conj(local_energy)).real()
                    );
                }
                else {
                    omega = exp(local_energy + conj(log_psi_prime - log_psi));
                    generic_atomicAdd(
                        this_.next_state_norm_avg,
                        weight * exp(2 * local_energy.real())
                    );
                }
                probability_ratio = exp(2.0 * (log_psi_prime.real() - log_psi.real()));

                generic_atomicAdd(this_.omega_avg, weight * omega);
                generic_atomicAdd(this_.probability_ratio_avg, weight * probability_ratio);
            }

            if(compute_gradient) {
                if(real_gradient) {
                    psi_prime_kernel.psi_real.foreach_O_k(
                        spins,
                        angles_prime,
                        [&](const unsigned int k, const double& O_k_element) {
                            generic_atomicAdd(&this_.omega_O_k_avg[k], weight * omega * O_k_element);
                            generic_atomicAdd(&this_.probability_ratio_O_k_avg[k], complex_t(weight * probability_ratio * O_k_element, 0.0));
                        }
                    );
                } else {
                    psi_prime_kernel.psi_imag.foreach_O_k(
                        spins,
                        angles_prime,
                        [&](const unsigned int k, const double& O_k_element) {
                            generic_atomicAdd(&this_.omega_O_k_avg[k], weight * omega * O_k_element);
                        }
                    );
                }
            }
        },
        max(psi.get_width(), psi_prime.get_width())
    );
}

} // namespace kernel


template<typename Psi_t, typename SpinEnsemble>
double HilbertSpaceDistance::gradient(
    complex<double>* result, const Psi_t& psi, const PsiPair& psi_prime, const Operator& operator_,
    const bool is_unitary, const SpinEnsemble& spin_ensemble
) {
    this->clear();
    this->compute_averages2<true, true>(psi, psi_prime, operator_, is_unitary, spin_ensemble);

    this->omega_avg_ar.update_host();
    this->omega_O_k_avg_ar.update_host();
    this->probability_ratio_avg_ar.update_host();
    this->probability_ratio_O_k_avg_ar.update_host();
    this->next_state_norm_avg_ar.update_host();

    this->omega_avg_ar.front() /= spin_ensemble.get_num_steps();
    this->probability_ratio_avg_ar.front() /= spin_ensemble.get_num_steps();
    this->next_state_norm_avg_ar.front() /= spin_ensemble.get_num_steps();

    auto u = (this->omega_avg_ar.front() * conj(this->omega_avg_ar.front())).real();
    auto v = this->next_state_norm_avg_ar.front() * this->probability_ratio_avg_ar.front();
    auto distance = sqrt(1.0 - u / v);

    for(auto k = 0u; k < this->num_params; k++) {
        this->omega_O_k_avg_ar.at(k) *= 1.0 / spin_ensemble.get_num_steps();
        this->probability_ratio_O_k_avg_ar.at(k) *= 1.0 / spin_ensemble.get_num_steps();

        const auto u_k_prime = 2.0 * (conj(this->omega_avg_ar.front()) * this->omega_O_k_avg_ar[k]).real();
        const auto v_k_prime = 2.0 * (this->next_state_norm_avg_ar.front() * this->probability_ratio_O_k_avg_ar[k].real());

        result[k] = complex<double>(
            (
                -0.5 * (u_k_prime * v - u * v_k_prime) / (v * v)
            ) / distance,
            0.0
        );

    }

    this->clear();
    this->compute_averages2<true, false>(psi, psi_prime, operator_, is_unitary, spin_ensemble);

    this->omega_avg_ar.update_host();
    this->omega_O_k_avg_ar.update_host();
    this->probability_ratio_avg_ar.update_host();
    this->next_state_norm_avg_ar.update_host();

    this->omega_avg_ar.front() /= spin_ensemble.get_num_steps();
    this->probability_ratio_avg_ar.front() /= spin_ensemble.get_num_steps();
    this->next_state_norm_avg_ar.front() /= spin_ensemble.get_num_steps();

    u = (this->omega_avg_ar.front() * conj(this->omega_avg_ar.front())).real();
    v = this->next_state_norm_avg_ar.front() * this->probability_ratio_avg_ar.front();
    distance = sqrt(1.0 - u / v);

    for(auto k = 0u; k < this->num_params; k++) {
        this->omega_O_k_avg_ar.at(k) *= 1.0 / spin_ensemble.get_num_steps();

        const auto u_k_prime = 2.0 * (conj(this->omega_avg_ar.front()) * this->omega_O_k_avg_ar[k]).imag();

        result[k] += complex<double>(
            0.0,
            (
                -0.5 * u_k_prime / v
            ) / distance
        );

    }

    return distance;
    // return v;
}


#ifdef ENABLE_MONTE_CARLO

#ifdef ENABLE_PSI_PAIR

template double HilbertSpaceDistance::gradient(complex<double>* result, const PsiPair& psi, const PsiPair& psi_prime, const Operator& operator_, const bool is_unitary, const MonteCarloLoop& spin_ensemble);

#endif

#if defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_PAIR)

template double HilbertSpaceDistance::gradient(complex<double>* result, const PsiClassical& psi, const PsiPair& psi_prime, const Operator& operator_, const bool is_unitary, const MonteCarloLoop& spin_ensemble);

#endif
#endif

#ifdef ENABLE_EXACT_SUMMATION

#ifdef ENABLE_PSI_PAIR

template double HilbertSpaceDistance::gradient(complex<double>* result, const PsiPair& psi, const PsiPair& psi_prime, const Operator& operator_, const bool is_unitary, const ExactSummation& spin_ensemble);

#endif

#if defined(ENABLE_PSI_CLASSICAL) && defined(ENABLE_PSI_PAIR)

template double HilbertSpaceDistance::gradient(complex<double>* result, const PsiClassical& psi, const PsiPair& psi_prime, const Operator& operator_, const bool is_unitary, const ExactSummation& spin_ensemble);

#endif
#endif

} // namespace rbm_on_gpu
