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
  : gpu(gpu) {
    this->N = N;
    this->M = M;
    this->prefactor = 1.0f;

    this->num_active_params = N + M + N * M;
    using complex_t = std::complex<float>;

    std::vector<complex_t> a_host(N);
    std::vector<complex_t> b_host(M);
    std::vector<complex_t> W_host(N * M);
    std::vector<complex_t> n_host(M);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> random_real(-1.0f, 1.0f);

    for(auto i = 0u; i < N; i++) {
        a_host[i] = complex_t(0.0f, 0.0f);
    }
    for(auto j = 0u; j < M; j++) {
        b_host[j] = complex_t(noise * random_real(rng), noise * random_real(rng));
    }
    for(auto i = 0u; i < N; i++) {
        for(auto j = 0u; j < M; j++) {
            const auto idx = j * N + i;
            W_host[idx] = (
                (i == j % N ? complex_t(1.0f, 3.14 / 4.0f) : complex_t(0.0f, 0.0f)) +
                complex_t(noise * random_real(rng), noise * random_real(rng))
            );
        }
    }
    for(auto j = 0u; j < M; j++) {
        n_host[j] = complex_t(1.0f, 0.0f);
    }

    this->allocate_memory();
    this->update_params(a_host.data(), b_host.data(), W_host.data(), n_host.data());
}

Psi::Psi(const unsigned int N, const unsigned int M, const bool gpu)
  : gpu(gpu) {
    this->N = N;
    this->M = M;
    this->prefactor = 1.0f;
    this->num_active_params = N + M + N * M;

    this->allocate_memory();
}

Psi::Psi(
    const unsigned int N,
    const unsigned int M,
    const std::complex<float>* a_host,
    const std::complex<float>* b_host,
    const std::complex<float>* W_host,
    const std::complex<float>* n_host,
    const float prefactor,
    const bool gpu
) : gpu(gpu) {
    this->N = N;
    this->M = M;
    this->prefactor = prefactor;
    this->num_active_params = N + M + N * M;

    this->allocate_memory();
    this->update_params(a_host, b_host, W_host, n_host);
}

Psi::Psi(const Psi& other) : gpu(other.gpu) {
    this->N = other.N;
    this->M = other.M;
    this->prefactor = other.prefactor;
    this->num_active_params = other.num_active_params;

    this->allocate_memory();
    this->update_params(
        reinterpret_cast<complex<float>*>(other.a),
        reinterpret_cast<complex<float>*>(other.b),
        reinterpret_cast<complex<float>*>(other.W),
        reinterpret_cast<complex<float>*>(other.n),
        other.gpu
    );
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

        #ifdef __CUDA_ARCH__
        if(threadIdx.x == 0)
        #endif
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

void Psi::allocate_memory() {
    MALLOC(this->a, sizeof(complex_t) * this->N, this->gpu);
    MALLOC(this->b, sizeof(complex_t) * this->M, this->gpu);
    MALLOC(this->W, sizeof(complex_t) * this->N * this->M, this->gpu);
    MALLOC(this->n, sizeof(complex_t) * this->M, this->gpu);
}

void Psi::update_params(
    const std::complex<float>* a,
    const std::complex<float>* b,
    const std::complex<float>* W,
    const std::complex<float>* n,
    const bool ptr_on_gpu
) {
    MEMCPY(this->a, a, sizeof(complex_t) * this->N, this->gpu, ptr_on_gpu);
    MEMCPY(this->b, b, sizeof(complex_t) * this->M, this->gpu, ptr_on_gpu);
    MEMCPY(this->W, W, sizeof(complex_t) * this->N * this->M, this->gpu, ptr_on_gpu);
    MEMCPY(this->n, n, sizeof(complex_t) * this->M, this->gpu, ptr_on_gpu);
}

Psi::~Psi() noexcept(false) {
    FREE(this->a, this->gpu);
    FREE(this->b, this->gpu);
    FREE(this->W, this->gpu);
    FREE(this->n, this->gpu);
}

} // namespace rbm_on_gpu
