#ifdef ENABLE_PSI

#include "quantum_state/Psi.hpp"
#include "network_functions/PsiVector.hpp"
#include "network_functions/PsiNorm.hpp"
#include "network_functions/PsiOkVector.hpp"
#include "spin_ensembles/ExactSummation.hpp"

#include <complex>
#include <vector>
#include <random>
#include <cstring>


namespace rbm_on_gpu {

Psi::Psi(const unsigned int N, const unsigned int M, const int seed, const double noise, const bool free_quantum_axis, const bool gpu)
  : alpha_array(N, false), beta_array(N, false), b_array(M, gpu), W_array(N * M, gpu), free_quantum_axis(free_quantum_axis), gpu(gpu) {
    this->N = N;
    this->M = M;
    this->prefactor = 1.0;
    this->num_params = N + M + N * M;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> random_real(-1.0, 1.0);

    for(auto j = 0u; j < M; j++) {
        this->b_array[j] = complex_t(noise * random_real(rng), noise * random_real(rng));
    }
    for(auto i = 0u; i < N; i++) {
        this->alpha_array[i] = 0.0;
        this->beta_array[i] = 0.0;

        for(auto j = 0u; j < M; j++) {
            const auto idx = j * N + i;
            this->W_array[idx] = (
                (i == j % N ? complex_t(1.0, 3.14 / 4.0) : complex_t(0.0, 0.0)) +
                complex_t(noise * random_real(rng), noise * random_real(rng))
            );
        }
    }

    this->b_array.update_device();
    this->W_array.update_device();

    this->update_kernel();
}

Psi::Psi(const Psi& other)
    :
    alpha_array(other.alpha_array),
    beta_array(other.beta_array),
    b_array(other.b_array),
    W_array(other.W_array),
    free_quantum_axis(other.free_quantum_axis),
    gpu(other.gpu) {
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

void Psi::as_vector(complex<double>* result) const {
    psi_vector(result, *this);
}

double Psi::norm_function(const ExactSummation& exact_summation) const {
    return psi_norm(*this, exact_summation);
}

void Psi::O_k_vector(complex<double>* result, const Spins& spins) const {
    psi_O_k_vector(result, *this, spins);
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
