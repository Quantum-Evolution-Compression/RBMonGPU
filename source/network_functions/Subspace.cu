#include "network_functions/Subspace.hpp"
#include "spin_ensembles.hpp"
#include "quantum_states.hpp"


namespace rbm_on_gpu {


template<typename Psi_t, typename Psi_t_prime, typename SpinEnsemble>
complex_t Subspace::operator()(Array<complex_t>& a_vec, const Psi_t& psi, const Psi_t_prime& psi_prime, SpinEnsemble& spin_ensemble) {
    const auto gpu = psi.gpu;
    const auto psi_kernel = psi.get_kernel();
    const auto psi_prime_kernel = psi_prime.get_kernel();

    a_vec.update_device();
    auto a_vec_ptr = a_vec.data();

    Array<complex_t> distance_ar(1, gpu);
    distance_ar.clear();
    auto distance_ptr = distance_ar.data();

    const auto log_prefactor_ratio = log(psi_prime.prefactor / psi.prefactor);

    spin_ensemble.foreach(
        psi,
        [=] __device__ __host__ (
            const unsigned int spin_index,
            const Spins& spins,
            const complex_t& log_psi,
            typename Psi_t::Angles& angles,
            const double weight
        ) {
            #include "cuda_kernel_defines.h"

            SHARED typename Psi_t_prime::Angles angles_prime;
            SHARED complex_t log_psi_prime;
            psi_prime_kernel.log_psi_s(log_psi_prime, spins, angles_prime);

            SINGLE {
                generic_atomicAdd(distance_ptr, (log_prefactor_ratio + log_psi_prime - log_psi) * a_vec_ptr[spin_index]);
            }
        },
        max(psi.get_width(), psi_prime.get_width())
    );

    distance_ar.update_host();

    return distance_ar.front();// * (1.0 / spin_ensemble.get_num_steps());
}


template<typename Psi_t, typename Psi_t_prime, typename SpinEnsemble>
pair<Array<complex_t>, complex_t> Subspace::gradient(Array<complex_t>& a_vec, const Psi_t& psi, const Psi_t_prime& psi_prime, SpinEnsemble& spin_ensemble) {

    const auto gpu = psi.gpu;
    const auto psi_kernel = psi.get_kernel();
    const auto psi_prime_kernel = psi_prime.get_kernel();

    a_vec.update_device();
    auto a_vec_ptr = a_vec.data();

    Array<complex_t> distance_ar(1, gpu);
    distance_ar.clear();
    auto distance_ptr = distance_ar.data();

    Array<complex_t> derivative_ar(psi_prime.get_num_params(), gpu);
    derivative_ar.clear();
    auto derivative_ptr = derivative_ar.data();

    const auto log_prefactor_ratio = log(psi_prime.prefactor / psi.prefactor);
    // const auto log_prefactor_prime = log(psi_prime.prefactor);

    spin_ensemble.foreach(
        psi,
        [=] __device__ __host__ (
            const unsigned int spin_index,
            const Spins& spins,
            const complex_t& log_psi,
            typename Psi_t::Angles& angles,
            const double weight
        ) {
            #include "cuda_kernel_defines.h"

            SHARED typename Psi_t_prime::Angles angles_prime;
            SHARED complex_t log_psi_prime;
            psi_prime_kernel.log_psi_s(log_psi_prime, spins, angles_prime);

            SHARED complex_t a_s;
            SINGLE {
                a_s = a_vec_ptr[spin_index];
                generic_atomicAdd(distance_ptr, (/*log_prefactor_ratio +*/ log_psi_prime - log_psi) * a_s);
            }
            SYNC;

            psi_prime_kernel.foreach_O_k(
                spins,
                angles_prime,
                [&](const unsigned int k, const complex_t& O_k_element) {
                    generic_atomicAdd(
                        &derivative_ptr[k], conj(O_k_element * a_s)
                    );
                }
            );
        },
        max(psi.get_width(), psi_prime.get_width())
    );

    distance_ar.update_host();
    derivative_ar.update_host();

    // distance_ar.front() /= spin_ensemble.get_num_steps();
    for(auto k = 0u; k < psi_prime.get_num_params(); k++) {
        // derivative_ptr[k] *= 1.0 / spin_ensemble.get_num_steps();

        derivative_ptr[k] *= 2.0 * distance_ar.front();
    }

    return {derivative_ar, distance_ar.front()};
}



#ifdef ENABLE_EXACT_SUMMATION

// #ifdef ENABLE_PSI_EXACT

// template complex_t Subspace::operator()(
//     Array<complex_t>& a_vec, const PsiExact& psi, const PsiExact& psi_prime, ExactSummation& spin_ensemble
// );
// template Array<complex_t> Subspace::gradient(
//     Array<complex_t>& a_vec, const PsiExact& psi, const PsiExact& psi_prime, ExactSummation& spin_ensemble
// );

// #endif

#if defined(ENABLE_PSI_EXACT) && defined(ENABLE_PSI_DEEP)

template complex_t Subspace::operator()(
    Array<complex_t>& a_vec, const PsiExact& psi, const PsiDeep& psi_prime, ExactSummation& spin_ensemble
);
template pair<Array<complex_t>, complex_t> Subspace::gradient(
    Array<complex_t>& a_vec, const PsiExact& psi, const PsiDeep& psi_prime, ExactSummation& spin_ensemble
);

#endif

#endif


}  // namespace rbm_on_gpu
