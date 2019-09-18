#include "network_functions/PsiOkVector.hpp"
#include "quantum_state/Psi.hpp"
#include "quantum_state/PsiDynamical.hpp"
#include "quantum_state/PsiDeep.hpp"
#include "spin_ensembles/ExactSummation.hpp"
#include "spin_ensembles/MonteCarloLoop.hpp"
#include "types.h"

namespace rbm_on_gpu {


template<typename Psi_t>
void psi_O_k_vector(complex<float>* result, const Psi_t& psi, const Spins& spins) {
    complex_t* result_ptr;
    auto O_k_length = psi.get_num_params();
    auto psi_kernel = psi.get_kernel();

    MALLOC(result_ptr, sizeof(complex_t) * O_k_length, psi.gpu);

    const auto functor = [=] __host__ __device__ () {
        #include "cuda_kernel_defines.h"

        SHARED typename Psi_t::Angles angles;
        angles.init(psi_kernel, spins);

        psi_kernel.foreach_O_k(
            spins,
            angles,
            [&](const unsigned int k, const complex_t& O_k_element) {
                result_ptr[k] = O_k_element;
            }
        );
    };

    if(psi.gpu) {
        cuda_kernel<<<1, psi.get_width()>>>(functor);
    }
    else {
        functor();
    }

    MEMCPY_TO_HOST(result, result_ptr, sizeof(complex_t) * O_k_length, psi.gpu);
    FREE(result_ptr, psi.gpu);
}


template<typename Psi_t, typename SpinEnsemble>
void psi_O_k_vector(complex<float>* result, complex<float>* result_std, const Psi_t& psi, const SpinEnsemble& spin_ensemble) {
    const auto O_k_length = psi.get_num_params();
    const auto psi_kernel = psi.get_kernel();

    complex_t* result_device;
    complex_t* result2_device;

    MALLOC(result_device, sizeof(complex_t) * O_k_length, psi.gpu);
    MALLOC(result2_device, sizeof(complex_t) * O_k_length, psi.gpu);
    MEMSET(result_device, 0, sizeof(complex_t) * O_k_length, psi.gpu);
    MEMSET(result2_device, 0, sizeof(complex_t) * O_k_length, psi.gpu);

    spin_ensemble.foreach(
        psi,
        [=] __device__ __host__ (
            const unsigned int spin_index,
            const Spins spins,
            const complex_t log_psi,
            typename Psi_t::Angles& angles,
            const float weight
        ) {
            psi_kernel.foreach_O_k(
                spins,
                angles,
                [&](const unsigned int k, const complex_t& O_k_element) {
                    generic_atomicAdd(&result_device[k], weight * O_k_element);
                    const auto O_k_element2 = complex_t(
                        O_k_element.real() * O_k_element.real(), O_k_element.imag() * O_k_element.imag()
                    );
                    generic_atomicAdd(&result2_device[k], weight * O_k_element2);
                }
            );
        }
    );

    MEMCPY_TO_HOST(result, result_device, sizeof(complex_t) * O_k_length, psi.gpu);
    MEMCPY_TO_HOST(result_std, result2_device, sizeof(complex_t) * O_k_length, psi.gpu);
    FREE(result_device, psi.gpu);
    FREE(result2_device, psi.gpu);

    for(auto k = 0u; k < O_k_length; k++) {
        result[k] *= 1.0f / O_k_length;
        result_std[k] *= 1.0f / O_k_length;

        result_std[k] = result_std[k] - complex<float>(
            result[k].real() * result[k].real(), result[k].imag() * result[k].imag()
        );
    }
}


template void psi_O_k_vector(complex<float>* result, const Psi& psi, const Spins& spins);
template void psi_O_k_vector(complex<float>* result, const PsiDynamical& psi, const Spins& spins);
template void psi_O_k_vector(complex<float>* result, const PsiDeep& psi, const Spins& spins);


template void psi_O_k_vector(complex<float>* result, complex<float>* result_std, const Psi& psi, const ExactSummation& spin_ensemble);
template void psi_O_k_vector(complex<float>* result, complex<float>* result_std, const Psi& psi, const MonteCarloLoop& spin_ensemble);
template void psi_O_k_vector(complex<float>* result, complex<float>* result_std, const PsiDynamical& psi, const ExactSummation& spin_ensemble);
template void psi_O_k_vector(complex<float>* result, complex<float>* result_std, const PsiDynamical& psi, const MonteCarloLoop& spin_ensemble);
template void psi_O_k_vector(complex<float>* result, complex<float>* result_std, const PsiDeep& psi, const ExactSummation& spin_ensemble);
template void psi_O_k_vector(complex<float>* result, complex<float>* result_std, const PsiDeep& psi, const MonteCarloLoop& spin_ensemble);

} // namespace rbm_on_gpu
