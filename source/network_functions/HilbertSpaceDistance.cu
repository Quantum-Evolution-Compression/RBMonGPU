#include "network_functions/HilbertSpaceDistance.hpp"
#include "spin_ensembles/ExactSummation.hpp"
#include "spin_ensembles/MonteCarloLoop.hpp"
#include "quantum_state/Psi.hpp"
#include "quantum_state/PsiDeep.hpp"

#include <cstring>
#include <math.h>


namespace rbm_on_gpu {

namespace kernel {


template<bool compute_gradient, bool free_quantum_axis, typename Psi_t, typename SpinEnsemble>
void kernel::HilbertSpaceDistance::compute_averages(
    const Psi_t& psi, const Psi_t& psi_prime, const Operator& operator_,
    const bool is_unitary, const SpinEnsemble& spin_ensemble
) const {
    const auto num_params = psi_prime.get_num_params();

    MEMSET(this->omega_avg, 0, sizeof(complex_t), this->gpu)
    MEMSET(this->omega_O_k_avg, 0, sizeof(complex_t) * num_params, this->gpu)
    MEMSET(this->probability_ratio_avg, 0, sizeof(double), this->gpu)
    MEMSET(this->probability_ratio_O_k_avg, 0, sizeof(complex_t) * num_params, this->gpu)
    MEMSET(this->next_state_norm_avg, 0, sizeof(double), this->gpu);

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

            SHARED typename Psi_t::Angles angles_prime;
            angles_prime.init(psi_prime_kernel, spins);

            SHARED complex_t log_psi_prime;
            psi_prime_kernel.log_psi_s(log_psi_prime, spins, angles_prime);

            SHARED complex_t psi_i_ratio[MAX_SPINS];
            if(free_quantum_axis) {
                SHARED Spins spins_i;
                SHARED complex_t log_psi_prime_i;
                for(auto i = 0u; i < N; i++) {
                    SYNC;
                    SINGLE {
                        spins_i = spins.flip(i);
                    }
                    SYNC;

                    angles_prime.init(psi_prime_kernel, spins_i);
                    psi_prime_kernel.log_psi_s(log_psi_prime_i, spins_i, angles_prime);
                    SINGLE {
                        psi_i_ratio[i] = exp(log_psi_prime_i - log_psi_prime);
                    }
                }
                SYNC;

                MULTI(i, N) {
                    generic_atomicAdd(
                        &log_psi_prime,
                        (
                            this_.delta_alpha[i] * spins[i] * psi_i_ratio[i] +
                            this_.delta_beta[i] * complex_t(0.0, 1.0) * (
                                psi_i_ratio[i] * this_.cos_sum_alpha[i] -
                                spins[i] * this_.sin_sum_alpha[i]
                            )
                        )
                    );
                }
                SYNC;

                // TODO: optimize
                angles_prime.init(psi_prime_kernel, spins);
            }

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
                if(free_quantum_axis) {
                    MULTI(i, N) {
                        const auto O_alpha_i = spins[i] * psi_i_ratio[i];
                        generic_atomicAdd(&this_.omega_O_k_avg[i], weight * omega * conj(O_alpha_i));
                        generic_atomicAdd(&this_.probability_ratio_O_k_avg[i], weight * probability_ratio * 2.0 * conj(O_alpha_i));

                        const auto O_beta_i = complex_t(0.0, 1.0) * (
                            psi_i_ratio[i] * this_.cos_sum_alpha[i] -
                            spins[i] * this_.sin_sum_alpha[i]
                        );
                        generic_atomicAdd(&this_.omega_O_k_avg[N + i], weight * omega * conj(O_beta_i));
                        generic_atomicAdd(&this_.probability_ratio_O_k_avg[N + i], weight * probability_ratio * 2.0 * conj(O_beta_i));
                    }
                }

                psi_prime_kernel.foreach_O_k(
                    spins,
                    angles_prime,
                    [&](const unsigned int k, const complex_t& O_k_element) {
                        generic_atomicAdd(&this_.omega_O_k_avg[k], weight * omega * conj(O_k_element));
                        generic_atomicAdd(&this_.probability_ratio_O_k_avg[k], weight * probability_ratio * 2.0 * conj(O_k_element));
                    }
                );
            }
        },
        max(psi.get_width(), psi_prime.get_width())
    );
}

// template<typename Psi_t, typename SpinEnsemble>
// void kernel::HilbertSpaceDistance::overlap(
//     const Psi_t& psi, const Psi_t& psi_prime, const SpinEnsemble& spin_ensemble
// ) const {
//     MEMSET(this->omega_avg, 0, sizeof(complex_t), this->gpu)
//     MEMSET(this->probability_ratio_avg, 0, sizeof(double), this->gpu)

//     const auto this_ = *this;
//     const auto psi_kernel = psi.get_kernel();
//     const auto psi_prime_kernel = psi_prime.get_kernel();

//     spin_ensemble.foreach(
//         psi,
//         [=] __device__ __host__ (
//             const unsigned int spin_index,
//             const Spins spins,
//             const complex_t log_psi,
//             typename Psi_t::Angles& angles,
//             const double weight
//         ) {
//             #include "cuda_kernel_defines.h"

//             SHARED typename Psi_t::Angles angles_prime;
//             angles_prime.init(psi_prime_kernel, spins);

//             SHARED complex_t log_psi_prime;
//             psi_prime_kernel.log_psi_s(log_psi_prime, spins, angles_prime);

//             SHARED complex_t   omega;
//             SHARED double      probability_ratio;

//             SINGLE
//             {
//                 omega = exp(conj(log_psi_prime - log_psi));
//                 probability_ratio = exp(2.0 * (log_psi_prime.real() - log_psi.real()));

//                 generic_atomicAdd(this_.omega_avg, weight * omega);
//                 generic_atomicAdd(this_.probability_ratio_avg, weight * probability_ratio);
//             }
//         },
//         max(psi.get_num_angles(), psi_prime.get_num_angles())
//     );
// }

} // namespace kernel

HilbertSpaceDistance::HilbertSpaceDistance(const unsigned int N, const unsigned int num_params, const bool gpu)
      : omega_O_k_avg_host(num_params),
        probability_ratio_O_k_avg_host(num_params),
        delta_alpha_ar(N, gpu),
        delta_beta_ar(N, gpu),
        sin_sum_alpha_ar(N, gpu),
        cos_sum_alpha_ar(N, gpu) {
    this->gpu = gpu;

    MALLOC(this->omega_avg, sizeof(complex_t), this->gpu);
    MALLOC(this->omega_O_k_avg, sizeof(complex_t) * num_params, this->gpu);
    MALLOC(this->probability_ratio_avg, sizeof(double), this->gpu);
    MALLOC(this->probability_ratio_O_k_avg, sizeof(complex_t) * num_params, this->gpu);
    MALLOC(this->next_state_norm_avg, sizeof(double), this->gpu);

    this->delta_alpha = this->delta_alpha_ar.data();
    this->delta_beta = this->delta_beta_ar.data();
    this->sin_sum_alpha = this->sin_sum_alpha_ar.data();
    this->cos_sum_alpha = this->cos_sum_alpha_ar.data();
}

HilbertSpaceDistance::~HilbertSpaceDistance() noexcept(false) {
    FREE(this->omega_avg, this->gpu);
    FREE(this->omega_O_k_avg, this->gpu);
    FREE(this->probability_ratio_avg, this->gpu);
    FREE(this->probability_ratio_O_k_avg, this->gpu);
    FREE(this->next_state_norm_avg, this->gpu);
}

template<typename Psi_t>
void HilbertSpaceDistance::update_quaxis(const Psi_t& psi, const Psi_t& psi_prime) {
    for(auto i = 0u; i < psi.get_num_spins(); i++) {
        const auto delta_alpha = psi_prime.alpha_array[i] - psi.alpha_array[i];
        const auto sum_alpha = psi_prime.alpha_array[i] + psi.alpha_array[i];
        const auto delta_beta = psi_prime.beta_array[i] - psi.beta_array[i];
        this->delta_alpha_ar[i] = delta_alpha;
        this->delta_beta_ar[i] = delta_beta;
        this->sin_sum_alpha_ar[i] = sin(sum_alpha);
        this->cos_sum_alpha_ar[i] = cos(sum_alpha);
    }

    this->delta_alpha_ar.update_device();
    this->delta_beta_ar.update_device();
    this->sin_sum_alpha_ar.update_device();
    this->cos_sum_alpha_ar.update_device();
}

template<typename Psi_t, typename SpinEnsemble>
double HilbertSpaceDistance::distance(
    const Psi_t& psi, const Psi_t& psi_prime, const Operator& operator_, const bool is_unitary,
    const SpinEnsemble& spin_ensemble
) {
    if(psi.free_quantum_axis) {
        this->update_quaxis(psi, psi_prime);
        this->compute_averages<false, true>(psi, psi_prime, operator_, is_unitary, spin_ensemble);
    }
    else {
        this->compute_averages<false, false>(psi, psi_prime, operator_, is_unitary, spin_ensemble);
    }

    complex_t omega_avg_host;
    double probability_ratio_avg_host;
    double next_state_norm_avg_host;

    MEMCPY_TO_HOST(&omega_avg_host, this->omega_avg, sizeof(complex_t), this->gpu);
    MEMCPY_TO_HOST(&probability_ratio_avg_host, this->probability_ratio_avg, sizeof(double), this->gpu);
    MEMCPY_TO_HOST(&next_state_norm_avg_host, this->next_state_norm_avg, sizeof(double), this->gpu);

    omega_avg_host /= spin_ensemble.get_num_steps();
    probability_ratio_avg_host /= spin_ensemble.get_num_steps();
    next_state_norm_avg_host /= spin_ensemble.get_num_steps();

    return 1.0 - sqrt(
        (omega_avg_host * conj(omega_avg_host)).real() / (
            next_state_norm_avg_host * probability_ratio_avg_host
        )
    );
}

// template<typename Psi_t, typename SpinEnsemble>
// double HilbertSpaceDistance::overlap(
//     const Psi_t& psi, const Psi_t& psi_prime, const SpinEnsemble& spin_ensemble
// ) const {
//     this->overlap(psi, psi_prime, spin_ensemble);

//     complex_t omega_avg_host;
//     double probability_ratio_avg_host;

//     MEMCPY_TO_HOST(&omega_avg_host, this->omega_avg, sizeof(complex_t), this->gpu);
//     MEMCPY_TO_HOST(&probability_ratio_avg_host, this->probability_ratio_avg, sizeof(double), this->gpu);

//     omega_avg_host /= spin_ensemble.get_num_steps();
//     probability_ratio_avg_host /= spin_ensemble.get_num_steps();

//     return sqrt(
//         (omega_avg_host * conj(omega_avg_host)).real() / probability_ratio_avg_host
//     );
// }

template<typename Psi_t, typename SpinEnsemble>
double HilbertSpaceDistance::gradient(
    complex<double>* result, const Psi_t& psi, const Psi_t& psi_prime, const Operator& operator_,
    const bool is_unitary, const SpinEnsemble& spin_ensemble
) {
    if(psi.free_quantum_axis) {
        this->update_quaxis(psi, psi_prime);
        this->compute_averages<true, true>(psi, psi_prime, operator_, is_unitary, spin_ensemble);
    }
    else {
        this->compute_averages<true, false>(psi, psi_prime, operator_, is_unitary, spin_ensemble);
    }

    complex<double> omega_avg_host;
    double probability_ratio_avg_host;
    const auto num_params = psi_prime.get_num_params();
    double next_state_norm_avg_host;

    MEMCPY_TO_HOST(&omega_avg_host, this->omega_avg, sizeof(complex_t), this->gpu);
    MEMCPY_TO_HOST(this->omega_O_k_avg_host.data(), this->omega_O_k_avg, sizeof(complex_t) * num_params, this->gpu);
    MEMCPY_TO_HOST(&probability_ratio_avg_host, this->probability_ratio_avg, sizeof(double), this->gpu);
    MEMCPY_TO_HOST(this->probability_ratio_O_k_avg_host.data(), this->probability_ratio_O_k_avg, sizeof(complex_t) * num_params, this->gpu);
    MEMCPY_TO_HOST(&next_state_norm_avg_host, this->next_state_norm_avg, sizeof(double), this->gpu);

    omega_avg_host /= spin_ensemble.get_num_steps();
    probability_ratio_avg_host /= spin_ensemble.get_num_steps();
    next_state_norm_avg_host /= spin_ensemble.get_num_steps();

    const auto u = (omega_avg_host * conj(omega_avg_host)).real();
    const auto v = next_state_norm_avg_host * probability_ratio_avg_host;

    for(auto k = 0u; k < num_params; k++) {
        this->omega_O_k_avg_host.at(k) *= 1.0 / spin_ensemble.get_num_steps();
        this->probability_ratio_O_k_avg_host.at(k) *= 1.0 / spin_ensemble.get_num_steps();

        const auto u_k_prime = 2.0 * conj(omega_avg_host) * this->omega_O_k_avg_host[k];
        const auto v_k_prime = next_state_norm_avg_host * this->probability_ratio_O_k_avg_host[k];

        result[k] = -0.5 * (u_k_prime * v - u * v_k_prime) / (v * v);
    }

    return 1.0 - sqrt(u / v);
}



template double HilbertSpaceDistance::distance(
    const Psi& psi, const Psi& psi_prime, const Operator& operator_, const bool is_unitary,
    const ExactSummation& spin_ensemble
);
template double HilbertSpaceDistance::distance(
    const Psi& psi, const Psi& psi_prime, const Operator& operator_, const bool is_unitary,
    const MonteCarloLoop& spin_ensemble
);

template double HilbertSpaceDistance::distance(
    const PsiDeep& psi, const PsiDeep& psi_prime, const Operator& operator_, const bool is_unitary,
    const ExactSummation& spin_ensemble
);
template double HilbertSpaceDistance::distance(
    const PsiDeep& psi, const PsiDeep& psi_prime, const Operator& operator_, const bool is_unitary,
    const MonteCarloLoop& spin_ensemble
);

// template double HilbertSpaceDistance::overlap(
//     const Psi& psi, const Psi& psi_prime, const ExactSummation& spin_ensemble
// );
// template double HilbertSpaceDistance::overlap(
//     const Psi& psi, const Psi& psi_prime, const MonteCarloLoop& spin_ensemble
// );

// template double HilbertSpaceDistance::overlap(
//     const PsiDeep& psi, const PsiDeep& psi_prime, const ExactSummation& spin_ensemble
// );
// template double HilbertSpaceDistance::overlap(
//     const PsiDeep& psi, const PsiDeep& psi_prime, const MonteCarloLoop& spin_ensemble
// );

template double HilbertSpaceDistance::gradient(
    complex<double>* result, const Psi& psi, const Psi& psi_prime, const Operator& operator_,
    const bool is_unitary, const ExactSummation& spin_ensemble
);
template double HilbertSpaceDistance::gradient(
    complex<double>* result, const Psi& psi, const Psi& psi_prime, const Operator& operator_,
    const bool is_unitary, const MonteCarloLoop& spin_ensemble
);

template double HilbertSpaceDistance::gradient(
    complex<double>* result, const PsiDeep& psi, const PsiDeep& psi_prime, const Operator& operator_,
    const bool is_unitary, const ExactSummation& spin_ensemble
);
template double HilbertSpaceDistance::gradient(
    complex<double>* result, const PsiDeep& psi, const PsiDeep& psi_prime, const Operator& operator_,
    const bool is_unitary, const MonteCarloLoop& spin_ensemble
);

} // namespace rbm_on_gpu
