#include "network_functions/PsiOkVector.hpp"
#include "quantum_state/Psi.hpp"
#include "quantum_state/PsiDeep.hpp"
#include "spin_ensembles.hpp"
#include "types.h"

namespace rbm_on_gpu {


template<typename Psi_t, typename SpinEnsemble>
Array<complex_t> get_S_matrix(const Psi_t& psi, SpinEnsemble& spin_ensemble) {
    const auto O_k_length = psi.get_num_params();
    const auto psi_kernel = psi.get_kernel();

    Array<complex_t> result(O_k_length * O_k_length, psi.gpu);
    result.clear();

    complex_t* data = result.data();

    spin_ensemble.foreach(
        psi,
        [=] __device__ __host__ (
            const unsigned int spin_index,
            const Spins spins,
            const complex_t log_psi,
            typename Psi_t::Angles& angles,
            const double weight
        ) {
            psi_kernel.foreach_O_k(
                spins,
                angles,
                [&](const unsigned int k, const complex_t& O_k_element) {
                    psi_kernel.foreach_O_k(
                        spins,
                        angles,
                        [&](const unsigned int k_prime, const complex_t& O_k_prime_element) {
                            generic_atomicAdd(&data[k * O_k_length + k_prime], O_k_element * conj(O_k_prime_element));
                        }
                    );
                }
            );
        }
    );
    result.update_host();

    for(auto k = 0u; k < O_k_length; k++) {
        for(auto k_prime = 0u; k_prime < O_k_length; k_prime++) {
            result[k * O_k_length + k_prime] /= spin_ensemble.get_num_steps();
        }
    }

    return result;
}


#if defined(ENABLE_PSI) && defined(ENABLE_EXACT_SUMMATION)
template Array<complex_t> get_S_matrix(const Psi&, ExactSummation&);
#endif

} // namespace rbm_on_gpu
