#ifdef ENABLE_EXACT_SUMMATION


#include "network_functions/PsiNorm.hpp"
#include "quantum_states.hpp"
#include "spin_ensembles/ExactSummation.hpp"
#include "types.h"


namespace rbm_on_gpu {


template<typename Psi_t>
double psi_norm(const Psi_t& psi, ExactSummation& exact_summation) {
    double* result_ptr;
    MALLOC(result_ptr, sizeof(double), psi.gpu);
    MEMSET(result_ptr, 0, sizeof(double), psi.gpu);

    auto this_ = psi.get_kernel();

    exact_summation.foreach(
        psi,
        [=] __host__ __device__ (
            const unsigned int spin_index,
            const Spins spins,
            const complex_t log_psi,
            const typename Psi_t::Angles& angles,
            const double weight
        ) {
            #ifdef __CUDA_ARCH__
            if(threadIdx.x == 0)
            #endif
            {
                generic_atomicAdd(result_ptr, this_.probability_s(log_psi.real()));
            }
        }
    );

    double result;
    MEMCPY_TO_HOST(&result, result_ptr, sizeof(double), psi.gpu);
    FREE(result_ptr, psi.gpu);

    return sqrt(result);
}


#ifdef ENABLE_PSI
template double psi_norm(const Psi& psi, ExactSummation&);
#endif // ENABLE_PSI

#ifdef ENABLE_PSI_DEEP
template double psi_norm(const PsiDeep& psi, ExactSummation&);
#endif // ENABLE_PSI_DEEP

#ifdef ENABLE_PSI_PAIR
template double psi_norm(const PsiPair& psi, ExactSummation&);
#endif // ENABLE_PSI_PAIR

#ifdef ENABLE_PSI_EXACT
template double psi_norm(const PsiExact& psi, ExactSummation&);
#endif // ENABLE_PSI_EXACT

} // namespace rbm_on_gpu

#endif // ENABLE_EXACT_SUMMATION
