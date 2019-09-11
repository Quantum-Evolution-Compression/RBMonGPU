#pragma once

#include "quantum_state/psi_functions.hpp"
#include "quantum_state/PsiCache.hpp"
#include "spin_ensembles/ExactSummation.hpp"
#include "Spins.h"
#include "types.h"
#ifdef __CUDACC__
    #include "utils.kernel"
#endif
#include "cuda_complex.hpp"

#include <vector>
#include <complex>
#include <memory>
#include <cassert>

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pytensor.hpp"

    using namespace std::complex_literals;
#endif // __PYTHONCC__


namespace rbm_on_gpu {

namespace kernel {

class Psi {
public:
    unsigned int N;
    unsigned int M;

    static constexpr unsigned int  max_N = MAX_SPINS;
    static constexpr unsigned int  max_M = MAX_HIDDEN_SPINS;

    unsigned int    num_active_params;
    float          prefactor;

    complex_t* a;
    complex_t* b;
    complex_t* W;
    complex_t* n;

// #ifdef __CUDACC__
    using Angles = rbm_on_gpu::PsiAngles;
    using Derivatives = rbm_on_gpu::PsiDerivatives;

// #endif

public:

    HDINLINE
    complex_t angle(const unsigned int j, const Spins& spins) const {
        complex_t result = this->b[j];

        const auto W_j = &(this->W[j]);
        for(unsigned int i = 0; i < this->N; i++) {
            result += W_j[i * this->M] * spins[i];
        }

        return result;
    }

    HDINLINE
    complex_t log_psi_s(const Spins& spins) const {
        complex_t result(0.0f, 0.0f);
        for(unsigned int i = 0; i < this->N; i++) {
            result += this->a[i] * spins[i];
        }
        for(unsigned int j = 0; j < this->M; j++) {
            result += my_logcosh(this->angle(j, spins));
        }

        return result;
    }

#ifdef __CUDACC__

    HDINLINE
    void log_psi_s(complex_t& result, const Spins& spins, const Angles& angles) const {
        // CAUTION: 'result' has to be a shared variable.
        // j = threadIdx.x

        #ifdef __CUDA_ARCH__

        auto summand = complex_t(
            (threadIdx.x < this->N ? this->a[threadIdx.x] * spins[threadIdx.x] : complex_t(0.0f, 0.0f)) +
            (threadIdx.x < this->M ? my_logcosh(angles[threadIdx.x]) : complex_t(0.0f, 0.0f))
        );

        tree_sum(result, this->M, summand);

        #else

        result = complex_t(0.0f, 0.0f);
        for(auto i = 0u; i < this->N; i++) {
            result += this->a[i] * spins[i];
        }
        for(auto j = 0u; j < this->M; j++) {
            result += my_logcosh(angles[j]);
        }

        #endif
    }

    HDINLINE
    void log_psi_s_real(float& result, const Spins& spins, const Angles& angles) const {
        // CAUTION: 'result' has to be a shared variable.
        // j = threadIdx.x

        #ifdef __CUDA_ARCH__

        auto summand = float(
            (threadIdx.x < this->N ? this->a[threadIdx.x].real() * spins[threadIdx.x] : 0.0f) +
            (threadIdx.x < this->M ? my_logcosh(angles[threadIdx.x]).real() : 0.0f)
        );

        tree_sum(result, this->M, summand);

        #else

        result = 0.0f;
        for(auto i = 0u; i < this->N; i++) {
            result += this->a[i].real() * spins[i];
        }
        for(auto j = 0u; j < this->M; j++) {
            result += my_logcosh(angles[j]).real();
        }

        #endif
    }

    HDINLINE void flip_spin_of_jth_angle(
        const unsigned int j, const unsigned int position, const Spins& new_spins, Angles& angles
    ) const {
        if(j < this->get_num_angles()) {
            angles[j] += 2.0f * new_spins[position] * this->W[position * this->M + j];
        }
    }

    HDINLINE
    complex_t psi_s(const Spins& spins, const Angles& angles) const {
        #include "cuda_kernel_defines.h"

        SHARED complex_t log_psi;
        this->log_psi_s(log_psi, spins, angles);

        return exp(log(this->prefactor) + log_psi);
    }

#endif // __CUDACC__

    // complex<float> psi_s_std(const Spins& spins) const {
    //     return this->psi_s(spins).to_std();
    // }

    // HDINLINE
    // float probability_s(const Spins& spins, const Angles& angles) const {
    //     return exp(2.0f * (log(this->prefactor) + this->log_psi_s(spins, angles).real()));
    // }

    // float probability_s_py(const Spins& spins) const {
    //     return this->probability_s(spins);
    // }

    HDINLINE
    float probability_s(const float log_psi_s_real) const {
        return exp(2.0f * (log(this->prefactor) + log_psi_s_real));
    }

    HDINLINE
    unsigned int get_num_spins() const {
        return this->N;
    }

    HDINLINE
    unsigned int get_num_hidden_spins() const {
        return this->M;
    }

    HDINLINE
    unsigned int get_num_angles() const {
        return this->M;
    }

    HDINLINE
    static constexpr unsigned int get_max_spins() {
        return max_N;
    }

    HDINLINE
    static constexpr unsigned int get_max_hidden_spins() {
        return max_M;
    }

    HDINLINE
    static constexpr unsigned int get_max_angles() {
        return max_M;
    }

    HDINLINE
    unsigned int get_num_params() const {
        return this->num_active_params + this->M;
    }

    HDINLINE
    unsigned int get_num_active_params() const {
        return this->num_active_params;
    }

    HDINLINE
    static unsigned int get_num_params(const unsigned int N, const unsigned int M) {
        return N + M + N * M + M;
    }

#ifdef __CUDACC__

    HDINLINE
    complex_t get_O_k_element(
        const unsigned int k,
        const Spins& spins,
        const PsiDerivatives& psi_derivatives
    ) const {
        if(k < this->N) {
            return complex_t(spins[k], 0.0f);
        }

        const auto N_plus_M = this->N + this->M;
        if(k < N_plus_M) {
            const auto j = k - this->N;
            return psi_derivatives.tanh_angles[j];
        }

        const auto i = (k - N_plus_M) / this->M;
        const auto j = (k - N_plus_M) % this->M;
        return psi_derivatives.tanh_angles[j] * spins[i];
    }

    template<typename DerivativesType>
    HDINLINE
    complex_t get_O_k_element(
        const unsigned int k,
        const Spins& spins,
        const DerivativesType& psi_derivatives
    ) const {
        return this->get_O_k_element(k, spins, psi_derivatives);
    }

    template<typename Function>
    HDINLINE
    void foreach_O_k(const Spins& spins, const Derivatives& derivatives, Function function) const {
        #ifdef __CUDA_ARCH__
        for(auto k = threadIdx.x; k < this->num_active_params; k += blockDim.x)
        #else
        for(auto k = 0u; k < this->num_active_params; k++)
        #endif
        {
            function(k, this->get_O_k_element(k, spins, derivatives));
        }
    }

    Psi get_kernel() const {
        return *this;
    }

#endif // __CUDACC__
};

} // namespace kernel


class Psi : public kernel::Psi {
public:
    bool gpu;

public:
    Psi(const unsigned int N, const unsigned int M, const bool gpu=true);
    Psi(const unsigned int N, const unsigned int M, const int seed=0, const float noise=1e-4, const bool gpu=true);
    Psi(
        const unsigned int N,
        const unsigned int M,
        const std::complex<float>* a,
        const std::complex<float>* b,
        const std::complex<float>* W,
        const std::complex<float>* n,
        const float prefactor=1.0f,
        const bool gpu=true
    );

#ifdef __PYTHONCC__
    Psi(
        const xt::pytensor<std::complex<float>, 1u>& a,
        const xt::pytensor<std::complex<float>, 1u>& b,
        const xt::pytensor<std::complex<float>, 2u>& W,
        const xt::pytensor<std::complex<float>, 1u>& n,
        const float prefactor=1.0f,
        const bool gpu=true
    ) : gpu(gpu) {
        this->N = a.shape()[0];
        this->M = b.shape()[0];
        this->prefactor = prefactor;

        this->num_active_params = this->N + this->M + this->N * this->M;

        this->allocate_memory();
        this->update_params(
            a.data(), b.data(), W.data(), n.data()
        );
    }

    void update_params(
        const xt::pytensor<std::complex<float>, 1u>& a,
        const xt::pytensor<std::complex<float>, 1u>& b,
        const xt::pytensor<std::complex<float>, 2u>& W,
        const xt::pytensor<std::complex<float>, 1u>& n
    ) {
        this->update_params(
            a.data(), b.data(), W.data(), n.data()
        );
    }

    xt::pytensor<complex<float>, 1> as_vector_py() const {
        auto result = xt::pytensor<complex<float>, 1>(
            std::array<long int, 1>({static_cast<long int>(pow(2, this->N))})
        );
        this->as_vector(result.data());

        return result;
    }

    xt::pytensor<complex<float>, 1> O_k_vector_py(const Spins& spins) const {
        auto result = xt::pytensor<complex<float>, 1>(
            std::array<long int, 1>({static_cast<long int>(this->num_active_params)})
        );
        this->O_k_vector(result.data(), spins);

        return result;
    }

#endif
    Psi(const Psi& other);
    ~Psi() noexcept(false);

    inline bool on_gpu() const {
        return this->gpu;
    }

    void update_params(
        const std::complex<float>* a,
        const std::complex<float>* b,
        const std::complex<float>* W,
        const std::complex<float>* n,
        const bool ptr_on_gpu=false
    );

    void as_vector(complex<float>* result) const;
    void O_k_vector(complex<float>* result, const Spins& spins) const;
    float norm_function(const ExactSummation& exact_summation) const;
    std::complex<float> log_psi_s_std(const Spins& spins);

    unsigned int get_num_params_py() const {
        return this->get_num_params();
    }

private:
    void allocate_memory();
};

} // namespace rbm_on_gpu
