#include "network_functions/RenyiCorrelation.hpp"
#include "spin_ensembles.hpp"
#include "quantum_states.hpp"
#include "cuda_complex.hpp"
#include "types.h"

namespace rbm_on_gpu {


template<typename Psi_t>
double RenyiCorrelation::operator()(const Psi_t& psi, const Operator& U_A, SpecialMonteCarloLoop& spin_ensemble) {
    this->result_ar.clear();
    auto result = this->result_ar.data();

    const auto psi_kernel = psi.get_kernel();
    const auto U_A_kernel = U_A.get_kernel();
    const auto N_A = psi.get_num_spins() / 2;

    spin_ensemble.foreach(
        psi,
        [=] __device__ __host__ (
            const unsigned int spin_index,
            const Spins* spins,
            const complex_t* log_psi,
            const typename Psi_t::Angles& angles,
            const double weight
        ) {
            #include "cuda_kernel_defines.h"

            SHARED complex_t local_energy[2];

            U_A_kernel.local_energy(local_energy[0], psi_kernel, spins[0], log_psi[0], angles);
            U_A_kernel.local_energy(local_energy[1], psi_kernel, spins[1], log_psi[1], angles);

            SINGLE {
                const auto hamming_distance = spins[0].extract_first_n(N_A).hamming_distance(spins[1].extract_first_n(N_A));
                const auto hamming_sign = (hamming_distance & 1u) ? -1.0 : 1.0;
                generic_atomicAdd(result, weight * hamming_sign * abs2(local_energy[0]) * abs2(local_energy[1]));
            }
        }
    );

    this->result_ar.update_host();
    return this->result_ar.front() / spin_ensemble.get_num_steps();
}

template<typename Psi_t>
double RenyiCorrelation::operator()(const Psi_t& psi, const Operator& U_A, SpecialExactSummation& spin_ensemble) {
    this->result_ar.clear();
    auto result = this->result_ar.data();

    const auto psi_kernel = psi.get_kernel();
    const auto U_A_kernel = U_A.get_kernel();
    const auto N_A = psi.get_num_spins() / 2;

    spin_ensemble.foreach(
        psi,
        [=] __device__ __host__ (
            const unsigned int spin_index,
            const Spins* spins,
            const complex_t* log_psi,
            const typename Psi_t::Angles& angles,
            const double weight
        ) {
            #include "cuda_kernel_defines.h"

            SHARED complex_t local_energy[2];

            U_A_kernel.local_energy(local_energy[0], psi_kernel, spins[0], log_psi[0], angles);
            U_A_kernel.local_energy(local_energy[1], psi_kernel, spins[1], log_psi[1], angles);

            SINGLE {
                const auto hamming_distance = spins[0].extract_first_n(N_A).hamming_distance(spins[1].extract_first_n(N_A));
                const auto hamming_sign = (hamming_distance & 1u) ? -1.0 : 1.0;
                const auto hamming_weight = 1.0 / double(1u << hamming_distance);
                generic_atomicAdd(result, weight * hamming_weight * hamming_sign * abs2(local_energy[0]) * abs2(local_energy[1]));
            }
        }
    );

    this->result_ar.update_host();
    return this->result_ar.front() / spin_ensemble.get_num_steps();
}



#ifdef ENABLE_SPECIAL_MONTE_CARLO

#ifdef ENABLE_PSI_DEEP

template double RenyiCorrelation::operator()(const PsiDeep& psi, const Operator& U_A, SpecialMonteCarloLoop& spin_ensemble);

#endif // ENABLE_PSI_DEEP
#ifdef ENABLE_PSI_EXACT

template double RenyiCorrelation::operator()(const PsiExact& psi, const Operator& U_A, SpecialMonteCarloLoop& spin_ensemble);

#endif // ENABLE_PSI_EXACT


#endif // ENABLE_SPECIAL_MONTE_CARLO

#ifdef ENABLE_SPECIAL_EXACT_SUMMATION

#ifdef ENABLE_PSI_DEEP

template double RenyiCorrelation::operator()(const PsiDeep& psi, const Operator& U_A, SpecialExactSummation& spin_ensemble);

#endif // ENABLE_PSI_DEEP
#ifdef ENABLE_PSI_EXACT

template double RenyiCorrelation::operator()(const PsiExact& psi, const Operator& U_A, SpecialExactSummation& spin_ensemble);

#endif // ENABLE_PSI_EXACT

#endif // ENABLE_SPECIAL_EXACT_SUMMATION


}  // namespace rbm_on_gpu
