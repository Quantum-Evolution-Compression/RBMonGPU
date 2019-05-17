#pragma once

#include "quantum_state/Psi.hpp"
#include "quantum_state/psi_functions.hpp"
#include "quantum_state/PsiW3Cache.hpp"
#include "Spins.h"
#include "types.h"
#include "cuda_complex.hpp"

#include <vector>
#include <complex>
#include <memory>
#include <cassert>

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pytensor.hpp"

    using namespace std::complex_literals;

#else
    DINLINE int __abs(const int x) {
        return x >= 0 ? x : -x;
    }
#endif // __CUDACC__


namespace rbm_on_gpu {

namespace kernel {

class PsiW3 : public Psi {
public:
    unsigned int F;

    static constexpr unsigned int  max_F = MAX_F;

    complex_t* X;
    complex_t* Y;

#ifdef __CUDACC__
    using Cache = rbm_on_gpu::PsiW3Cache;
#endif

public:

#ifdef __CUDACC__

    HDINLINE
    void init_angles(complex_t* angles, const Spins& spins) const {

        // initialize f-angles

        #ifdef __CUDA_ARCH__
            const auto f = threadIdx.x;
            if(f < this->F)
        #else
            for(auto f = 0u; f < this->F; f++)
        #endif

        {
            angles[this->M + f] = complex_t(0.0, 0.0);
            for(auto i = 0u; i < this->N; i++) {
                angles[this->M + f] += this->X[i * this->F + f] * spins[i];
            }
        }

        #ifdef __CUDA_ARCH__
        __syncthreads();
        #endif

        // initialize j-angles

        #ifdef __CUDA_ARCH__
            const auto j = threadIdx.x;
            if(j < this->M)
        #else
            for(auto j = 0u; j < this->M; j++)
        #endif

        {
            angles[j] = this->b[j];

            const auto W_j = &(this->W[j]);
            for(auto i = 0u; i < this->N; i++) {
                angles[j] += W_j[i * this->M] * spins[i];
            }

            for(auto f = 0u; f < this->F; f++) {
                const auto f_angle = angles[this->M + f];
                angles[j] += f_angle * f_angle * this->Y[f * this->M + j];
            }
        }
    }

    HDINLINE complex_t flip_spin_of_jth_angle(
        const complex_t* angles, const unsigned int j, const unsigned int position, const Spins& new_spins
    ) const {
        // CAUTION: When calling from the CPU make sure that j < M is called before j >= M!
        // CAUTION: This function contains a thread-synchronization!

        // First, compute delta_f_total = 2 * X_f * delta_f + delta_f^2 for each f-angle

        #ifdef __CUDA_ARCH__
            __shared__ complex_t delta_f_total[max_F];
            const auto f = threadIdx.x;
            if(f < this->F)
        #else
            complex_t delta_f_total[max_F];
            for(auto f = 0u; f < this->F; f++)
        #endif

        {
            const auto delta_f = 2.0 * new_spins[position] * this->X[position * this->F + f];

            delta_f_total[f] = 2.0 * angles[this->M + f] * delta_f + delta_f * delta_f;
        }

        #ifdef __CUDA_ARCH__
        __syncthreads();
        #endif

        // -------------------------------------------------------
        if(j >= this->get_num_angles()) {
            return complex_t();
        }

        auto result = angles[j];

        // update j-angles
        if(j < this->M) {
            result += 2.0 * new_spins[position] * this->W[position * this->M + j];

            for(auto f = 0u; f < this->F; f++) {
                result += delta_f_total[f] * this->Y[f * this->M + j];
            }
        }
        // update f-angles
        else {
            const auto f = j - this->M;
            result += 2.0 * new_spins[position] * this->X[position * this->F + f];
        }

        return result;
    }

    HDINLINE
    complex_t get_O_k_element(
        const unsigned int k,
        const Spins& spins,
        const complex_t* Z_f,
        const complex_t* f_angles,
        const complex_t* tanh_j_angles
    ) const {
        // a_i
        if(k < this->N) {
            return complex_t(spins[k], 0.0);
        }
        auto offset = this->N;

        // b_j
        if(k < offset + this->M) {
            const auto j = k - offset;

            return tanh_j_angles[j];
        }
        offset += this->M;

        // W_ij
        if(k < offset + this->N * this->M) {
            const auto i = (k - offset) / this->M;
            const auto j = (k - offset) % this->M;

            #ifdef __CUDA_ARCH__
            if(min(__abs(i - j), N - __abs(i - j)) > 3)
            #else
            if(min(abs(i - j), N - abs(i - j)) > 3)
            #endif
            {
                return complex_t(0.0, 0.0);
            }
            else {
                return tanh_j_angles[j] * spins[i];
            }
        }
        offset += this->N * this->M;

        // X_if
        if(k < offset + this->N * this->F) {
            const auto i = (k - offset) / this->F;
            const auto f = (k - offset) % this->F;

            return 40.0 * Z_f[f] * spins[i];
        }
        offset += this->N * this->F;

        // Y_fj
        const auto f = (k - offset) / this->M;
        const auto j = (k - offset) % this->M;

        return 0.0 * tanh_j_angles[j] * f_angles[f] * f_angles[f];
    }

    HDINLINE
    complex_t get_O_k_element(
        const unsigned int k,
        const Spins& spins,
        const Cache& psi_cache
    ) const {
        return this->get_O_k_element(k, spins, psi_cache.Z_f, psi_cache.f_angles, psi_cache.tanh_j_angles);
    }

    template<typename Function>
    HDINLINE
    void foreach_O_k(const Spins& spins, const Cache& cache, Function function) const {
        #ifdef __CUDA_ARCH__
        for(auto k = threadIdx.x; k < this->num_active_params; k += blockDim.x)
        #else
        for(auto k = 0u; k < this->num_active_params; k++)
        #endif
        {
            function(k, this->get_O_k_element(k, spins, cache));
        }
    }
#endif // __CUDACC__

    HDINLINE
    unsigned int get_num_angles() const {
        return this->M + this->F;
    }

    HDINLINE
    static constexpr unsigned int get_max_angles() {
        return max_M + max_F;
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
    static unsigned int get_num_params(const unsigned int N, const unsigned int M, const unsigned int F) {
        return N + M + N * M + M + N * F + F * M;
    }
};

} // namespace kernel


class PsiW3 : public kernel::PsiW3 {
public:
    bool gpu;

public:
    PsiW3(const unsigned int N, const unsigned int M, const unsigned int F, const bool gpu=true);
    PsiW3(const unsigned int N, const unsigned int M, const unsigned int F, const int seed=0, const double noise=1e-4, const bool gpu=true);
    PsiW3(
        const unsigned int N,
        const unsigned int M,
        const unsigned int F,
        const std::complex<double>* a,
        const std::complex<double>* b,
        const std::complex<double>* W,
        const std::complex<double>* n,
        const std::complex<double>* X,
        const std::complex<double>* Y,
        const double prefactor=1.0,
        const bool gpu=true
    );

#ifdef __PYTHONCC__
    PsiW3(
        const xt::pytensor<std::complex<double>, 1u>& a,
        const xt::pytensor<std::complex<double>, 1u>& b,
        const xt::pytensor<std::complex<double>, 2u>& W,
        const xt::pytensor<std::complex<double>, 1u>& n,
        const xt::pytensor<std::complex<double>, 2u>& X,
        const xt::pytensor<std::complex<double>, 2u>& Y,
        const double prefactor=1.0,
        const bool gpu=true
    ) : gpu(gpu) {
        this->N = a.shape()[0];
        this->M = b.shape()[0];
        this->F = X.shape()[1];
        this->prefactor = prefactor;

        this->num_active_params = this->N + this->M + this->N * this->M + this->N * this->F + this->F * this->M;

        this->allocate_memory();
        this->update_params(
            a.raw_data(), b.raw_data(), W.raw_data(), n.raw_data(), X.raw_data(), Y.raw_data()
        );
    }

    void update_params(
        const xt::pytensor<std::complex<double>, 1u>& a,
        const xt::pytensor<std::complex<double>, 1u>& b,
        const xt::pytensor<std::complex<double>, 2u>& W,
        const xt::pytensor<std::complex<double>, 1u>& n,
        const xt::pytensor<std::complex<double>, 2u>& X,
        const xt::pytensor<std::complex<double>, 2u>& Y
    ) {
        this->update_params(
            a.raw_data(), b.raw_data(), W.raw_data(), n.raw_data(), X.raw_data(), Y.raw_data()
        );
    }

    xt::pytensor<complex<double>, 1> as_vector_py() const {
        auto result = xt::pytensor<complex<double>, 1>(
            std::array<long int, 1>({static_cast<long int>(pow(2, this->N))})
        );
        this->as_vector(result.raw_data());

        return result;
    }

    xt::pytensor<complex<double>, 1> O_k_vector_py(const Spins& spins) const {
        auto result = xt::pytensor<complex<double>, 1>(
            std::array<long int, 1>({static_cast<long int>(this->num_active_params)})
        );
        this->O_k_vector(result.raw_data(), spins);

        return result;
    }

#endif // __CUDACC__
    PsiW3(const PsiW3& other);
    ~PsiW3() noexcept(false);

    inline bool on_gpu() const {
        return this->gpu;
    }

    void update_params(
        const std::complex<double>* a,
        const std::complex<double>* b,
        const std::complex<double>* W,
        const std::complex<double>* n,
        const std::complex<double>* X,
        const std::complex<double>* Y,
        const bool ptr_on_gpu=false
    );

    inline kernel::PsiW3 get_kernel() const {
        return static_cast<kernel::PsiW3>(*this);
    }

    void as_vector(complex<double>* result) const;
    double norm_function() const;
    void O_k_vector(complex<double>* result, const Spins& spins) const;

    unsigned int get_num_params_py() const {
        return this->get_num_params();
    }

private:
    void allocate_memory();
};

} // namespace rbm_on_gpu
