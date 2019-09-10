#include "network_functions/PsiVector.hpp"
#include "quantum_state/Psi.hpp"
#include "quantum_state/PsiDynamical.hpp"
#include "spin_ensembles/ExactSummation.hpp"
#include "types.h"

namespace rbm_on_gpu {


template<typename Psi_t>
void psi_vector(complex<double>* result, const Psi_t& psi) {
    ExactSummation exact_summation(psi.N, psi.gpu);

    complex_t* result_ptr;
    MALLOC(result_ptr, sizeof(complex_t) * exact_summation.get_num_steps(), psi.gpu);

    const auto log_prefactor = log(psi.prefactor);
    auto psi_kernel = psi.get_kernel();

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
                result_ptr[spin_index] = exp(log_prefactor + log_psi);
            }
        }
    );

    MEMCPY_TO_HOST(result, result_ptr, sizeof(complex_t) * exact_summation.get_num_steps(), psi.gpu);
    FREE(result_ptr, psi.gpu);
}


template void psi_vector(complex<double>* result, const Psi& psi);
template void psi_vector(complex<double>* result, const PsiDynamical& psi);

} // namespace rbm_on_gpu
