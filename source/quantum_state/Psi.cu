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

Psi::Psi(const unsigned int N, const unsigned int M, const int seed, const float noise, const bool gpu)
  : a_array(N, gpu), b_array(M, gpu), W_array(N * M, gpu), gpu(gpu) {
    this->N = N;
    this->M = M;
    this->prefactor = 1.0f;
    this->num_params = N + M + N * M;
    this->a = this->a_array.data();
    this->b = this->b_array.data();
    this->W = this->W_array.data();

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> random_real(-1.0f, 1.0f);

    for(auto i = 0u; i < N; i++) {
        this->a_array[i] = complex_t(0.0f, 0.0f);
    }
    for(auto j = 0u; j < M; j++) {
        this->b_array[j] = complex_t(noise * random_real(rng), noise * random_real(rng));
    }
    for(auto i = 0u; i < N; i++) {
        for(auto j = 0u; j < M; j++) {
            const auto idx = j * N + i;
            this->W_array[idx] = (
                (i == j % N ? complex_t(1.0f, 3.14 / 4.0f) : complex_t(0.0f, 0.0f)) +
                complex_t(noise * random_real(rng), noise * random_real(rng))
            );
        }
    }

    this->update_kernel();
    this->create_index_pairs();
}

Psi::Psi(
    const unsigned int N,
    const unsigned int M,
    const std::complex<float>* a_host,
    const std::complex<float>* b_host,
    const std::complex<float>* W_host,
    const float prefactor,
    const bool gpu
) : a_array(N, gpu), b_array(M, gpu), W_array(N * M, gpu), gpu(gpu) {
    cout << "constr" << endl;
    this->N = N;
    this->M = M;
    this->prefactor = prefactor;
    this->num_params = N + M + N * M;

    memcpy(this->a_array.host_data(), a_host, sizeof(complex_t) * N);
    memcpy(this->b_array.host_data(), b_host, sizeof(complex_t) * M);
    memcpy(this->W_array.host_data(), W_host, sizeof(complex_t) * N * M);

    this->update_kernel();
    this->create_index_pairs();
}

Psi::Psi(const Psi& other)
    :
    a_array(other.N, other.gpu),
    b_array(other.M, other.gpu),
    W_array(other.N * other.M, other.gpu),
    gpu(gpu) {
    cout << "copy _" << other.gpu << endl;
    this->N = other.N;
    this->M = other.M;
    this->prefactor = other.prefactor;
    this->num_params = other.num_params;

    this->a_array = other.a_array;
    this->b_array = other.b_array;
    this->W_array = other.W_array;

    cout << "copy 1" << endl;

    this->update_kernel();
    cout << "copy 2" << endl;
    this->create_index_pairs();
    cout << "copy 3" << endl;
}

Psi::~Psi() noexcept(false) {
    cout << "destr" << endl;
}

void Psi::update_kernel() {
    // Note: This does not only update the memory on the device, but also the pointer which can be host or device

    this->a = this->a_array.data();
    this->b = this->b_array.data();
    this->W = this->W_array.data();

    this->a_array.update_device();
    this->b_array.update_device();
    this->W_array.update_device();
}

void Psi::as_vector(complex<float>* result) const {
    psi_vector(result, *this);
}

float Psi::norm_function(const ExactSummation& exact_summation) const {
    return psi_norm(*this, exact_summation);
}

void Psi::O_k_vector(complex<float>* result, const Spins& spins) const {
    psi_O_k_vector(result, *this, spins);
}

std::complex<float> Psi::log_psi_s_std(const Spins& spins) {
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

void Psi::get_params(complex<float>* result) const {
    memcpy(result, this->a_array.data(), sizeof(complex_t) * this->N);
    memcpy(result + this->N, this->b_array.data(), sizeof(complex_t) * this->M);
    memcpy(result + this->N + this->M, this->W_array.data(), sizeof(complex_t) * this->N * this->M);
}

void Psi::set_params(const complex<float>* new_params) {
    memcpy(this->a_array.data(), new_params, sizeof(complex_t) * this->N);
    memcpy(this->b_array.data(), new_params + this->N, sizeof(complex_t) * this->M);
    memcpy(this->W_array.data(), new_params + this->N + this->M, sizeof(complex_t) * this->N * this->M);

    this->update_kernel();
}

} // namespace rbm_on_gpu
