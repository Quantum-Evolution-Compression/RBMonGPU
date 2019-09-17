#include "network_functions/HilbertSpaceDistance.hpp"
#include "spin_ensembles/ExactSummation.hpp"
#include "spin_ensembles/MonteCarloLoop.hpp"
#include "quantum_state/PsiDynamical.hpp"
#include "quantum_state/Psi.hpp"

#include <cstring>


namespace rbm_on_gpu {

namespace kernel {


template<bool compute_gradient, typename Psi_t, typename SpinEnsemble>
void kernel::HilbertSpaceDistance::compute_averages(
    const Psi_t& psi, const Psi_t& psi_prime, const Operator& operator_,
    const bool is_unitary, const SpinEnsemble& spin_ensemble
) const {
    const auto O_k_length = psi_prime.get_num_params();

    MEMSET(this->omega_avg, 0, sizeof(complex_t), this->gpu)
    MEMSET(this->omega_O_k_avg, 0, sizeof(complex_t) * O_k_length, this->gpu)
    MEMSET(this->probability_ratio_avg, 0, sizeof(float), this->gpu)
    MEMSET(this->probability_ratio_O_k_avg, 0, sizeof(complex_t) * O_k_length, this->gpu)
    MEMSET(this->next_state_norm_avg, 0, sizeof(float), this->gpu);

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
            const float weight
        ) {
            #include "cuda_kernel_defines.h"

            SHARED complex_t local_energy;
            operator_.local_energy(local_energy, psi_kernel, spins, log_psi, angles);

            SHARED typename Psi_t::Angles angles_prime;
            angles_prime.init(psi_prime_kernel, spins);

            SHARED complex_t log_psi_prime;
            psi_prime_kernel.log_psi_s(log_psi_prime, spins, angles_prime);

            SHARED complex_t   omega;
            SHARED float      probability_ratio;

            #ifdef __CUDA_ARCH__
            if(threadIdx.x == 0)
            #endif
            {
                if(is_unitary) {
                    omega = exp(conj(log_psi_prime - log_psi)) * local_energy;
                    // TODO: move to `record`
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
                probability_ratio = exp(2.0f * (log_psi_prime.real() - log_psi.real()));

                generic_atomicAdd(this_.omega_avg, weight * omega);
                generic_atomicAdd(this_.probability_ratio_avg, weight * probability_ratio);
            }

            if(compute_gradient) {
                psi_prime_kernel.foreach_O_k(
                    spins,
                    angles_prime,
                    [&](const unsigned int k, const complex_t& O_k_element) {
                        generic_atomicAdd(&this_.omega_O_k_avg[k], weight * omega * conj(O_k_element));
                        generic_atomicAdd(&this_.probability_ratio_O_k_avg[k], weight * probability_ratio * 2.0f * conj(O_k_element));
                    }
                );
            }
        },
        max(psi.get_num_angles(), psi_prime.get_num_angles())
    );
}

} // namespace kernel

HilbertSpaceDistance::HilbertSpaceDistance(const unsigned int O_k_length, const bool gpu)
      : omega_O_k_avg_host(O_k_length),
        probability_ratio_O_k_avg_host(O_k_length) {
    this->gpu = gpu;

    MALLOC(this->omega_avg, sizeof(complex_t), this->gpu);
    MALLOC(this->omega_O_k_avg, sizeof(complex_t) * O_k_length, this->gpu);
    MALLOC(this->probability_ratio_avg, sizeof(float), this->gpu);
    MALLOC(this->probability_ratio_O_k_avg, sizeof(complex_t) * O_k_length, this->gpu);
    MALLOC(this->next_state_norm_avg, sizeof(float), this->gpu);

    this->local_energies = nullptr;
    this->num_local_energies = 0u;
}

HilbertSpaceDistance::~HilbertSpaceDistance() noexcept(false) {
    FREE(this->omega_avg, this->gpu);
    FREE(this->omega_O_k_avg, this->gpu);
    FREE(this->probability_ratio_avg, this->gpu);
    FREE(this->probability_ratio_O_k_avg, this->gpu);
    FREE(this->next_state_norm_avg, this->gpu);

    FREE(this->local_energies, this->gpu);
}

void HilbertSpaceDistance::allocate_local_energies(const unsigned int num_local_energies) {
    if(num_local_energies <= this->num_local_energies) {
        return;
    }
    this->num_local_energies = num_local_energies;

    FREE(this->local_energies, this->gpu);
    MALLOC(this->local_energies, sizeof(complex_t) * this->num_local_energies, this->gpu);
}

template<typename Psi_t, typename SpinEnsemble>
float HilbertSpaceDistance::distance(
    const Psi_t& psi, const Psi_t& psi_prime, const Operator& operator_, const bool is_unitary,
    const SpinEnsemble& spin_ensemble
) const {
    this->compute_averages<false>(psi, psi_prime, operator_, is_unitary, spin_ensemble);

    complex_t omega_avg_host;
    float probability_ratio_avg_host;
    float next_state_norm_avg_host;

    MEMCPY_TO_HOST(&omega_avg_host, this->omega_avg, sizeof(complex_t), this->gpu);
    MEMCPY_TO_HOST(&probability_ratio_avg_host, this->probability_ratio_avg, sizeof(float), this->gpu);
    MEMCPY_TO_HOST(&next_state_norm_avg_host, this->next_state_norm_avg, sizeof(float), this->gpu);

    omega_avg_host /= spin_ensemble.get_num_steps();
    probability_ratio_avg_host /= spin_ensemble.get_num_steps();
    next_state_norm_avg_host /= spin_ensemble.get_num_steps();

    return 1.0f - sqrt(
        (omega_avg_host * conj(omega_avg_host)).real() / (
            next_state_norm_avg_host * probability_ratio_avg_host
        )
    );
}

template<typename Psi_t, typename SpinEnsemble>
float HilbertSpaceDistance::gradient(
    complex<float>* result, const Psi_t& psi, const Psi_t& psi_prime, const Operator& operator_,
    const bool is_unitary, const SpinEnsemble& spin_ensemble
) {
    this->compute_averages<true>(psi, psi_prime, operator_, is_unitary, spin_ensemble);

    complex<float> omega_avg_host;
    float probability_ratio_avg_host;
    const auto O_k_length = psi_prime.get_num_params();
    float next_state_norm_avg_host;

    MEMCPY_TO_HOST(&omega_avg_host, this->omega_avg, sizeof(complex_t), this->gpu);
    MEMCPY_TO_HOST(this->omega_O_k_avg_host.data(), this->omega_O_k_avg, sizeof(complex_t) * O_k_length, this->gpu);
    MEMCPY_TO_HOST(&probability_ratio_avg_host, this->probability_ratio_avg, sizeof(float), this->gpu);
    MEMCPY_TO_HOST(this->probability_ratio_O_k_avg_host.data(), this->probability_ratio_O_k_avg, sizeof(complex_t) * O_k_length, this->gpu);
    MEMCPY_TO_HOST(&next_state_norm_avg_host, this->next_state_norm_avg, sizeof(float), this->gpu);

    omega_avg_host /= spin_ensemble.get_num_steps();
    probability_ratio_avg_host /= spin_ensemble.get_num_steps();
    next_state_norm_avg_host /= spin_ensemble.get_num_steps();

    const auto u = (omega_avg_host * conj(omega_avg_host)).real();
    const auto v = next_state_norm_avg_host * probability_ratio_avg_host;

    for(auto k = 0u; k < O_k_length; k++) {
        this->omega_O_k_avg_host.at(k) *= 1.0f / spin_ensemble.get_num_steps();
        this->probability_ratio_O_k_avg_host.at(k) *= 1.0f / spin_ensemble.get_num_steps();

        const auto u_k_prime = 2.0f * conj(omega_avg_host) * this->omega_O_k_avg_host[k];
        const auto v_k_prime = next_state_norm_avg_host * this->probability_ratio_O_k_avg_host[k];

        result[k] = -0.5f * (u_k_prime * v - u * v_k_prime) / (v * v);
    }

    return 1.0f - sqrt(u / v);
}



template float HilbertSpaceDistance::distance(
    const Psi& psi, const Psi& psi_prime, const Operator& operator_, const bool is_unitary,
    const ExactSummation& spin_ensemble
) const;
template float HilbertSpaceDistance::distance(
    const Psi& psi, const Psi& psi_prime, const Operator& operator_, const bool is_unitary,
    const MonteCarloLoop& spin_ensemble
) const;

template float HilbertSpaceDistance::distance(
    const PsiDynamical& psi, const PsiDynamical& psi_prime, const Operator& operator_, const bool is_unitary,
    const ExactSummation& spin_ensemble
) const;
template float HilbertSpaceDistance::distance(
    const PsiDynamical& psi, const PsiDynamical& psi_prime, const Operator& operator_, const bool is_unitary,
    const MonteCarloLoop& spin_ensemble
) const;


template float HilbertSpaceDistance::gradient(
    complex<float>* result, const Psi& psi, const Psi& psi_prime, const Operator& operator_,
    const bool is_unitary, const ExactSummation& spin_ensemble
);
template float HilbertSpaceDistance::gradient(
    complex<float>* result, const Psi& psi, const Psi& psi_prime, const Operator& operator_,
    const bool is_unitary, const MonteCarloLoop& spin_ensemble
);

template float HilbertSpaceDistance::gradient(
    complex<float>* result, const PsiDynamical& psi, const PsiDynamical& psi_prime, const Operator& operator_,
    const bool is_unitary, const ExactSummation& spin_ensemble
);
template float HilbertSpaceDistance::gradient(
    complex<float>* result, const PsiDynamical& psi, const PsiDynamical& psi_prime, const Operator& operator_,
    const bool is_unitary, const MonteCarloLoop& spin_ensemble
);

} // namespace rbm_on_gpu
