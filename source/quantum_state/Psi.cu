#ifdef ENABLE_PSI

#include "quantum_state/Psi.hpp"
#include "network_functions/PsiVector.hpp"
#include "network_functions/PsiNorm.hpp"
#include "network_functions/PsiOkVector.hpp"

#include <complex>
#include <vector>
#include <random>
#include <cstring>


namespace rbm_on_gpu {


Psi::Psi(const Psi& other)
    :
    rbm_on_gpu::PsiBase(other),
    b_array(other.b_array),
    W_array(other.W_array)
{
    this->N = other.N;
    this->M = other.M;
    this->prefactor = other.prefactor;
    this->num_params = other.num_params;

    this->update_kernel();
}

void Psi::update_kernel() {
    this->b = this->b_array.data();
    this->W = this->W_array.data();
}

std::complex<double> Psi::log_psi_s_std(const Spins& spins) {
    complex_t* result;
    MALLOC(result, sizeof(complex_t), this->gpu);

    auto this_ = this->get_kernel();

    const auto functor = [=] __host__ __device__ () {
        #include "cuda_kernel_defines.h"

        SHARED Psi::Angles angles;
        angles.init(this_, spins);

        SHARED complex_t log_psi;
        this_.log_psi_s(log_psi, spins, angles);

        SINGLE
        {
            *result = log_psi;
        }
    };

    if(this->gpu) {
        cuda_kernel<<<1, this->get_num_angles()>>>(functor);
    }
    else {
        functor();
    }

    complex_t result_host;
    MEMCPY_TO_HOST(&result_host, result, sizeof(complex_t), this->gpu);
    FREE(result, this->gpu);

    return result_host.to_std();
}

void Psi::get_params(complex<double>* result) const {
    for(auto i = 0u; i < this->N; i++) {
        result[i] = complex<double>(this->alpha_array[i], 0.0);
        result[this->N + i] = complex<double>(this->beta_array[i], 0.0);
    }
    memcpy(result + 2 * this->N, this->b_array.host_data(), sizeof(complex_t) * this->M);
    memcpy(result + 2 * this->N + this->M, this->W_array.host_data(), sizeof(complex_t) * this->N * this->M);
}

void Psi::set_params(const complex<double>* new_params) {
    for(auto i = 0u; i < this->N; i++) {
        this->alpha_array[i] = new_params[i].real();
        this->beta_array[i] = new_params[this->N + i].real();
    }
    memcpy(this->b_array.host_data(), new_params + 2 * this->N, sizeof(complex_t) * this->M);
    memcpy(this->W_array.host_data(), new_params + 2 * this->N + this->M, sizeof(complex_t) * this->N * this->M);

    this->b_array.update_device();
    this->W_array.update_device();

    this->update_kernel();
}

} // namespace rbm_on_gpu


#endif // ENABLE_PSI
