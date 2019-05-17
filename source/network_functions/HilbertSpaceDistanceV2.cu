#include "network_functions/HilbertSpaceDistanceV2.hpp"
#include "spin_ensembles/ExactSummation.hpp"
#include "spin_ensembles/MonteCarloLoop.hpp"
#include "quantum_state/Psi.hpp"

#include <cstring>


namespace rbm_on_gpu {

namespace kernel {

template<bool compute_gradient, typename SpinEnsemble>
void kernel::HilbertSpaceDistanceV2::compute_averages(
    const rbm_on_gpu::Psi& psi, const Psi& psi_prime, const Operator& operator_,
    const bool is_unitary, const SpinEnsemble& spin_ensemble
) const {
    MEMSET(this->A_avg, 0, sizeof(complex_t), this->gpu)
    MEMSET(this->B_avg, 0, sizeof(complex_t), this->gpu)
    MEMSET(this->AB_avg, 0, sizeof(complex_t), this->gpu)
    MEMSET(this->A2_re_avg, 0, sizeof(double), this->gpu)
    MEMSET(this->A2_im_avg, 0, sizeof(double), this->gpu)
    MEMSET(this->B2_re_avg, 0, sizeof(double), this->gpu)
    MEMSET(this->B2_im_avg, 0, sizeof(double), this->gpu)

    if(compute_gradient) {
        MEMSET(this->O_k_avg, 0, sizeof(complex_t), this->gpu);
        MEMSET(this->A_O_k_avg, 0, sizeof(complex_t), this->gpu);
        MEMSET(this->B_O_k_re_avg, 0, sizeof(double), this->gpu);
        MEMSET(this->B_O_k_im_avg, 0, sizeof(double), this->gpu);
    }

    const auto O_k_length = psi.get_num_params();
    const auto this_ = *this;
    const auto psi_kernel = psi.get_kernel();

    spin_ensemble.foreach(
        psi,
        [=] __device__ __host__ (
            const unsigned int spin_index,
            const Spins spins,
            const complex_t log_psi,
            const complex_t* angle_ptr,
            const double weight
        ) {
            #ifdef __CUDA_ARCH__
            #define SHARED __shared__
            #else
            #define SHARED
            #endif

            SHARED complex_t local_energy;
            operator_.local_energy(local_energy, psi_kernel, spins, log_psi, angle_ptr);

            #ifdef __CUDA_ARCH__

            const auto angle_prime = psi_prime.angle(threadIdx.x, spins);
            const auto angle_prime_ptr = &angle_prime;

            #else

            complex_t angle_prime_ptr[MAX_HIDDEN_SPINS];
            for(auto j = 0u; j < psi_kernel.get_num_angles(); j++) {
                angle_prime_ptr[j] = psi_prime.angle(j, spins);
            }

            #endif

            SHARED complex_t log_psi_prime;
            psi_prime.log_psi_s(log_psi_prime, spins, angle_prime_ptr);

            const auto A = local_energy;
            const auto B = log_psi_prime - log_psi;

            #ifdef __CUDA_ARCH__
            if(threadIdx.x == 0)
            #endif
            {
                generic_atomicAdd(this_.A_avg, weight * A);
                generic_atomicAdd(this_.B_avg, weight * B);
                generic_atomicAdd(this_.AB_avg, weight * A * conj(B));
                generic_atomicAdd(this_.A2_re_avg, weight * A.real() * A.real());
                generic_atomicAdd(this_.A2_im_avg, weight * A.imag() * A.imag());
                generic_atomicAdd(this_.B2_re_avg, weight * B.real() * B.real());
                generic_atomicAdd(this_.B2_im_avg, weight * B.imag() * B.imag());
            }

            if(compute_gradient) {
                SHARED PsiCache psi_prime_cache;
                #ifdef __CUDA_ARCH__
                if(threadIdx.x < psi_prime.get_num_angles())
                #endif
                {
                    psi_prime_cache.init(angle_prime_ptr, psi_prime);
                }

                #ifdef __CUDA_ARCH__
                __syncthreads();
                for(auto k = threadIdx.x; k < O_k_length; k += blockDim.x)
                #else
                for(auto k = 0u; k < O_k_length; k++)
                #endif
                {
                    const auto O_k_element = psi_prime.get_O_k_element(k, spins, psi_prime_cache);

                    generic_atomicAdd(&this_.O_k_avg[k], weight * O_k_element);
                    generic_atomicAdd(&this_.A_O_k_avg[k], weight * A * conj(O_k_element));
                    generic_atomicAdd(&this_.B_O_k_re_avg[k], weight * B.real() * O_k_element.real());
                    generic_atomicAdd(&this_.B_O_k_im_avg[k], weight * B.imag() * O_k_element.imag());
                }
            }
        },
        compute_gradient ? 256 : psi.get_num_angles()
    );
}

} // namespace kernel

HilbertSpaceDistanceV2::HilbertSpaceDistanceV2(const bool gpu) {
    this->gpu = gpu;

    MALLOC(this->A_avg, sizeof(complex_t), this->gpu);
    MALLOC(this->B_avg, sizeof(complex_t), this->gpu);
    MALLOC(this->AB_avg, sizeof(complex_t), this->gpu);
    MALLOC(this->A2_re_avg, sizeof(double), this->gpu);
    MALLOC(this->A2_im_avg, sizeof(double), this->gpu);
    MALLOC(this->B2_re_avg, sizeof(double), this->gpu);
    MALLOC(this->B2_im_avg, sizeof(double), this->gpu);
}

HilbertSpaceDistanceV2::~HilbertSpaceDistanceV2() noexcept(false) {
    FREE(this->A_avg, this->gpu);
    FREE(this->B_avg, this->gpu);
    FREE(this->AB_avg, this->gpu);
    FREE(this->A2_re_avg, this->gpu);
    FREE(this->A2_im_avg, this->gpu);
    FREE(this->B2_re_avg, this->gpu);
    FREE(this->B2_im_avg, this->gpu);
}

template<typename SpinEnsemble>
double HilbertSpaceDistanceV2::distance(
    const Psi& psi, const Psi& psi_prime, const Operator& operator_, const bool is_unitary,
    const SpinEnsemble& spin_ensemble
) const {
    this->compute_averages<false>(psi, psi_prime, operator_, is_unitary, spin_ensemble);

    complex_t A;
    complex_t B;
    complex_t AB;
    double A2_re;
    double A2_im;
    double B2_re;
    double B2_im;

    MEMCPY_TO_HOST(&A, this->A_avg, sizeof(complex_t), this->gpu);
    MEMCPY_TO_HOST(&B, this->B_avg, sizeof(complex_t), this->gpu);
    MEMCPY_TO_HOST(&AB, this->AB_avg, sizeof(complex_t), this->gpu);
    MEMCPY_TO_HOST(&A2_re, this->A2_re_avg, sizeof(double), this->gpu);
    MEMCPY_TO_HOST(&A2_im, this->A2_im_avg, sizeof(double), this->gpu);
    MEMCPY_TO_HOST(&B2_re, this->B2_re_avg, sizeof(double), this->gpu);
    MEMCPY_TO_HOST(&B2_im, this->B2_im_avg, sizeof(double), this->gpu);

    A *= 1.0 / spin_ensemble.get_num_steps();
    B *= 1.0 / spin_ensemble.get_num_steps();
    AB *= 1.0 / spin_ensemble.get_num_steps();
    A2_re *= 1.0 / spin_ensemble.get_num_steps();
    A2_im *= 1.0 / spin_ensemble.get_num_steps();
    B2_re *= 1.0 / spin_ensemble.get_num_steps();
    B2_im *= 1.0 / spin_ensemble.get_num_steps();

    const auto deltaA_2_re = A2_re - A.real() * A.real();
    const auto deltaA_2_im = A2_im - A.imag() * A.imag();
    const auto deltaB_2_re = B2_re - B.real() * B.real();
    const auto deltaB_2_im = B2_im - B.imag() * B.imag();

    return -1.0 / 2 * (
        -deltaA_2_re - deltaA_2_im - deltaB_2_re - deltaB_2_im +
        2.0 * (AB - A * conj(B)).real()
    );
}


template double HilbertSpaceDistanceV2::distance(
    const Psi& psi, const Psi& psi_prime, const Operator& operator_, const bool is_unitary,
    const ExactSummation& spin_ensemble
) const;
template double HilbertSpaceDistanceV2::distance(
    const Psi& psi, const Psi& psi_prime, const Operator& operator_, const bool is_unitary,
    const MonteCarloLoop& spin_ensemble
) const;

} // namespace rbm_on_gpu
