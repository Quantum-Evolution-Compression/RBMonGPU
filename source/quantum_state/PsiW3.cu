#include "quantum_state/PsiW3.hpp"
#include "network_functions/PsiVector.hpp"
#include "network_functions/PsiNorm.hpp"
#include "network_functions/PsiOkVector.hpp"

#include <complex>
#include <vector>
#include <random>
#include <cstring>


namespace rbm_on_gpu {

PsiW3::PsiW3(const unsigned int N, const unsigned int M, const unsigned int F, const int seed, const double noise, const bool gpu)
  : gpu(gpu) {
    this->N = N;
    this->M = M;
    this->F = F;
    this->prefactor = 1.0;

    this->num_active_params = this->N + this->M + this->N * this->M + this->N * this->F + this->F * this->M;
    using complex_t = std::complex<double>;

    std::vector<complex_t> a_host(N);
    std::vector<complex_t> b_host(M);
    std::vector<complex_t> W_host(N * M);
    std::vector<complex_t> n_host(M);
    std::vector<complex_t> X_host(N * F);
    std::vector<complex_t> Y_host(F * M);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> random_real(-1.0, 1.0);

    for(auto i = 0u; i < N; i++) {
        a_host[i] = complex_t(0.0, 0.0);
    }
    for(auto j = 0u; j < M; j++) {
        b_host[j] = complex_t(noise * random_real(rng), noise * random_real(rng));
    }
    for(auto i = 0u; i < N; i++) {
        for(auto j = 0u; j < M; j++) {
            W_host[j * N + i] = (
                (i == j % N ? complex_t(0.01, 3.14 / 4.0) : complex_t(0.0, 0.0)) +
                complex_t(noise * random_real(rng), noise * random_real(rng))
            );
        }
    }
    for(auto j = 0u; j < M; j++) {
        n_host[j] = complex_t(1.0, 0.0);
    }
    for(auto i = 0u; i < N; i++) {
        for(auto f = 0u; f < F; f++) {
            X_host[i * this->F + f] = (
                (i == f % N ? complex_t(1.0, 0.0) : complex_t(0.0, 0.0)) +
                complex_t(noise * random_real(rng), noise * random_real(rng))
            );
        }
    }
    for(auto f = 0u; f < F; f++) {
        for(auto j = 0u; j < M; j++) {
            Y_host[f * this->M + j] = (
                complex_t(noise * random_real(rng), noise * random_real(rng))
            );
        }
    }

    this->allocate_memory();
    this->update_params(a_host.data(), b_host.data(), W_host.data(), n_host.data(), X_host.data(), Y_host.data());
}

PsiW3::PsiW3(const unsigned int N, const unsigned int M, const unsigned int F, const bool gpu)
  : gpu(gpu) {
    this->N = N;
    this->M = M;
    this->F = F;
    this->prefactor = 1.0;
    this->num_active_params = this->N + this->M + this->N * this->M + this->N * this->F + this->F * this->M;

    this->allocate_memory();
}

PsiW3::PsiW3(
    const unsigned int N,
    const unsigned int M,
    const unsigned int F,
    const std::complex<double>* a_host,
    const std::complex<double>* b_host,
    const std::complex<double>* W_host,
    const std::complex<double>* n_host,
    const std::complex<double>* X_host,
    const std::complex<double>* Y_host,
    const double prefactor,
    const bool gpu
) : gpu(gpu) {
    this->N = N;
    this->M = M;
    this->F = F;
    this->prefactor = prefactor;
    this->num_active_params = this->N + this->M + this->N * this->M + this->N * this->F + this->F * this->M;

    this->allocate_memory();
    this->update_params(a_host, b_host, W_host, n_host, X_host, Y_host);
}

PsiW3::PsiW3(const PsiW3& other) : gpu(other.gpu) {
    this->N = other.N;
    this->M = other.M;
    this->F = other.F;
    this->prefactor = other.prefactor;
    this->num_active_params = other.num_active_params;

    this->allocate_memory();
    this->update_params(
        reinterpret_cast<complex<double>*>(other.a),
        reinterpret_cast<complex<double>*>(other.b),
        reinterpret_cast<complex<double>*>(other.W),
        reinterpret_cast<complex<double>*>(other.n),
        reinterpret_cast<complex<double>*>(other.X),
        reinterpret_cast<complex<double>*>(other.Y),
        other.gpu
    );
}

void PsiW3::as_vector(complex<double>* result) const {
    psi_vector(result, *this);
}

double PsiW3::norm_function() const {
    return psi_norm(*this);
}

void PsiW3::O_k_vector(complex<double>* result, const Spins& spins) const {
    psi_O_k_vector(result, *this, spins);
}

void PsiW3::allocate_memory() {
    MALLOC(this->a, sizeof(complex_t) * this->N, this->gpu);
    MALLOC(this->b, sizeof(complex_t) * this->M, this->gpu);
    MALLOC(this->W, sizeof(complex_t) * this->N * this->M, this->gpu);
    MALLOC(this->n, sizeof(complex_t) * this->M, this->gpu);
    MALLOC(this->X, sizeof(complex_t) * this->N * this->F, this->gpu);
    MALLOC(this->Y, sizeof(complex_t) * this->F * this->M, this->gpu);
}

void PsiW3::update_params(
    const std::complex<double>* a,
    const std::complex<double>* b,
    const std::complex<double>* W,
    const std::complex<double>* n,
    const std::complex<double>* X,
    const std::complex<double>* Y,
    const bool ptr_on_gpu
) {
    MEMCPY(this->a, a, sizeof(complex_t) * this->N, this->gpu, ptr_on_gpu);
    MEMCPY(this->b, b, sizeof(complex_t) * this->M, this->gpu, ptr_on_gpu);
    MEMCPY(this->W, W, sizeof(complex_t) * this->N * this->M, this->gpu, ptr_on_gpu);
    MEMCPY(this->n, n, sizeof(complex_t) * this->M, this->gpu, ptr_on_gpu);
    MEMCPY(this->X, X, sizeof(complex_t) * this->N * this->F, this->gpu, ptr_on_gpu);
    MEMCPY(this->Y, Y, sizeof(complex_t) * this->F * this->M, this->gpu, ptr_on_gpu);
}

PsiW3::~PsiW3() noexcept(false) {
    FREE(this->a, this->gpu);
    FREE(this->b, this->gpu);
    FREE(this->W, this->gpu);
    FREE(this->n, this->gpu);
    FREE(this->X, this->gpu);
    FREE(this->Y, this->gpu);
}

} // namespace rbm_on_gpu
