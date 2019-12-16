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

Psi::Psi(const unsigned int N, const unsigned int M, const int seed, const double noise, const bool gpu)
  : a_array(N, gpu), alpha_array(N, gpu), b_array(M, gpu), W_array(N * M, gpu), gpu(gpu) {
    this->N = N;
    this->M = M;
    this->prefactor = 1.0;
    this->num_params = N + M + N * M;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> random_real(-1.0, 1.0);

    for(auto i = 0u; i < N; i++) {
        this->a_array[i] = complex_t(0.0, 0.0);
    }
    for(auto j = 0u; j < M; j++) {
        this->b_array[j] = complex_t(noise * random_real(rng), noise * random_real(rng));
    }
    for(auto i = 0u; i < N; i++) {
        for(auto j = 0u; j < M; j++) {
            const auto idx = j * N + i;
            this->W_array[idx] = (
                (i == j % N ? complex_t(1.0, 3.14 / 4.0) : complex_t(0.0, 0.0)) +
                complex_t(noise * random_real(rng), noise * random_real(rng))
            );
        }
    }

    this->a_array.update_device();
    this->b_array.update_device();
    this->W_array.update_device();

    this->update_kernel();
    this->create_index_pairs();
}

Psi::Psi(const Psi& other)
    :
    a_array(other.a_array),
    alpha_array(other.alpha_array),
    b_array(other.b_array),
    W_array(other.W_array),
    gpu(other.gpu) {
    this->N = other.N;
    this->M = other.M;
    this->prefactor = other.prefactor;
    this->num_params = other.num_params;

    this->update_kernel();
    this->create_index_pairs();
}

void Psi::update_kernel() {
    this->a = this->a_array.data();
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

void Psi::create_index_pairs() {
    for(auto i = 0; i < this->N; i++) {
        this->index_pair_list.push_back(make_pair(i, -1));
    }
    for(auto j = 0; j < this->M; j++) {
        this->index_pair_list.push_back(make_pair(-1, j));
    }
    for(auto i = 0; i < this->N; i++) {
        for(auto j = 0; j < this->M; j++) {
            this->index_pair_list.push_back(make_pair(i, j));
        }
    }
}

void Psi::get_params(complex<double>* result) const {
    memcpy(result, this->a_array.host_data(), sizeof(complex_t) * this->N);
    memcpy(result + this->N, this->b_array.host_data(), sizeof(complex_t) * this->M);
    memcpy(result + this->N + this->M, this->W_array.host_data(), sizeof(complex_t) * this->N * this->M);
}

void Psi::set_params(const complex<double>* new_params) {
    memcpy(this->a_array.host_data(), new_params, sizeof(complex_t) * this->N);
    memcpy(this->b_array.host_data(), new_params + this->N, sizeof(complex_t) * this->M);
    memcpy(this->W_array.host_data(), new_params + this->N + this->M, sizeof(complex_t) * this->N * this->M);

    this->a_array.update_device();
    this->b_array.update_device();
    this->W_array.update_device();

    this->update_kernel();
}

} // namespace rbm_on_gpu
