#ifdef ENABLE_EXACT_SUMMATION


#include "network_functions/PsiVector.hpp"
#include "quantum_states.hpp"
#include "spin_ensembles/ExactSummation.hpp"
#include "types.h"


namespace rbm_on_gpu {


template<typename Psi_t>
void psi_vector(complex<double>* result, const Psi_t& psi) {
    ExactSummation exact_summation(psi.get_num_spins(), psi.gpu);

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

template<typename Psi_t>
Array<complex_t> psi_vector(const Psi_t& psi) {
    Array<complex_t> result(1 << psi.get_num_spins(), false);
    psi_vector(reinterpret_cast<complex<double>*>(result.data()), psi);

    return result;
}


#ifdef ENABLE_PSI
template void psi_vector(complex<double>* result, const Psi& psi);
template Array<complex_t> psi_vector(const Psi& psi);
#endif // ENABLE_PSI

#ifdef ENABLE_PSI_DEEP
template void psi_vector(complex<double>* result, const PsiDeep& psi);
template Array<complex_t> psi_vector(const PsiDeep& psi);
#endif // ENABLE_PSI_DEEP

#ifdef ENABLE_PSI_PAIR
template void psi_vector(complex<double>* result, const PsiPair& psi);
template Array<complex_t> psi_vector(const PsiPair& psi);
#endif // ENABLE_PSI_PAIR

#ifdef ENABLE_PSI_CLASSICAL
template void psi_vector(complex<double>* result, const PsiClassical& psi);
template Array<complex_t> psi_vector(const PsiClassical& psi);
#endif // ENABLE_PSI_CLASSICAL

#ifdef ENABLE_PSI_EXACT
template void psi_vector(complex<double>* result, const PsiExact& psi);
template Array<complex_t> psi_vector(const PsiExact& psi);
#endif // ENABLE_PSI_EXACT

} // namespace rbm_on_gpu


#endif // ENABLE_EXACT_SUMMATION
