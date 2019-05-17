#pragma once

#include "spin_ensembles/SpinHistory.hpp"
#include "quantum_state/PsiHistory.hpp"
#include "quantum_state/PsiCache.hpp"
#include "quantum_state/Psi.hpp"
#include "operator/Operator.hpp"

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pytensor.hpp"
#endif // __CUDACC__

#include <complex>
// #include <memory>
// #include <array>

namespace rbm_on_gpu {

namespace kernel {

class DifferentiatePsi {
protected:
    unsigned int    O_k_length;
    unsigned int    num_spins;
    unsigned int    num_hidden_spins;

    rbm_on_gpu::SpinHistory*    spin_history_ptr;
    rbm_on_gpu::PsiHistory*     psi_history_ptr;

    kernel::Psi     psi;

    complex_t*  forces;
    complex_t*  O_k_avg;
    complex_t*  E_local_avg;
    complex_t*  E_local_O_k_avg;

    complex_t*  x;
    complex_t*  Ax;
    complex_t*  O_k_avg_x;

    complex<double>* forces_host;

    friend void evaluate_Ax(int*, std::complex<double>* x, std::complex<double>* Ax);
public:

#ifdef __CUDACC__

    HDINLINE
    void compute_averages(
        const unsigned int step,
        const Spins& spins,
        const complex_t& log_psi,
        const complex_t* angle_ptr,
        const double weight,
        const kernel::Operator operator_,
        const kernel::PsiHistory& psi_history
    ) const {
        #ifdef __CUDA_ARCH__

        __shared__ complex_t local_energy;
        operator_.local_energy(local_energy, this->psi, spins, log_psi, angle_ptr);

        __shared__ PsiCache psi_cache;

        if(threadIdx.x < this->num_hidden_spins) {
            psi_cache.init(angle_ptr, this->psi);
            psi_cache.store(step, psi_history);
        }

        __syncthreads();

        if(threadIdx.x == 0) {
            atomicAdd(this->E_local_avg, weight * local_energy);
        }

        for(auto k = threadIdx.x; k < this->O_k_length; k += blockDim.x) {
            const auto O_k_element = this->psi.get_O_k_element(k, spins, psi_cache);

            atomicAdd(&this->O_k_avg[k], weight * O_k_element);
            atomicAdd(&this->E_local_O_k_avg[k], weight * local_energy * conj(O_k_element));
        }

        #else

        complex_t local_energy;
        operator_.local_energy(local_energy, this->psi, spins, log_psi, angle_ptr);

        PsiCache psi_cache(angle_ptr, this->psi);
        psi_cache.store(step, psi_history);

        *this->E_local_avg += weight * local_energy;

        for(auto k = 0u; k < this->O_k_length; k++) {
            const auto O_k_element = this->psi.get_O_k_element(k, spins, psi_cache);

            this->O_k_avg[k] += weight * O_k_element;
            this->E_local_O_k_avg[k] += weight * local_energy * conj(O_k_element);
        }

        #endif
    }

    HDINLINE
    void evaluate_first_part_of_Ax(
        const unsigned int step,
        const kernel::SpinHistory& spin_history,
        const kernel::PsiHistory& psi_history
    ) const {
        #ifdef __CUDA_ARCH__

        __shared__ Spins     spins;
        __shared__ complex_t O_k_x;
        __shared__ PsiCache  psi_cache;

        if(threadIdx.x < this->num_hidden_spins) {
            psi_cache.load(step, psi_history);
            if(threadIdx.x == 0) {
                spins = spin_history.spins[step];
                O_k_x = complex_t(0.0, 0.0);
            }
        }

        __syncthreads();

        const auto effective_O_k_length = this->O_k_length - (this->O_k_length % blockDim.x) + blockDim.x;

        for(auto k = threadIdx.x; k < effective_O_k_length; k += blockDim.x) {
            const auto O_k_element = (
                k < this->O_k_length ?
                    this->psi.get_O_k_element(k, spins, psi_cache) :
                    complex_t(0.0, 0.0)
            );

            auto summand = O_k_element * this->x[k];
            for (int offset = 16; offset > 0; offset /= 2) {
                summand.__re_ += __shfl_down_sync(0xffffffff, summand.real(), offset);
                summand.__im_ += __shfl_down_sync(0xffffffff, summand.imag(), offset);
            }

            if(threadIdx.x % 32u == 0) {
                atomicAdd(&O_k_x, summand);
            }
        }

        __syncthreads();

        // TODO: decide this at compile-time
        const double weight = spin_history.weights != nullptr ? spin_history.weights[step] : 1.0;

        for(unsigned int k = threadIdx.x; k < this->O_k_length; k += blockDim.x) {
            const auto O_k_element = this->psi.get_O_k_element(k, spins, psi_cache);

            atomicAdd(&this->Ax[k], weight * conj(O_k_element) * O_k_x);
        }

        #else

        Spins       spins = spin_history.spins[step];
        complex_t   O_k_x(0.0, 0.0);
        PsiCache    psi_cache(step, psi_history);

        for(auto k = 0u; k < this->O_k_length; k ++) {
            const auto O_k_element = this->psi.get_O_k_element(k, spins, psi_cache);

            O_k_x += O_k_element * this->x[k];
        }

        // TODO: decide this at compile-time
        const double weight = spin_history.weights != nullptr ? spin_history.weights[step] : 1.0;

        for(auto k = 0u; k < this->O_k_length; k++) {
            const auto O_k_element = this->psi.get_O_k_element(k, spins, psi_cache);

            this->Ax[k] += weight * conj(O_k_element) * O_k_x;
        }

        #endif
    }

    HDINLINE
    void compute_forces() const {
        #ifdef __CUDA_ARCH__

        const auto k = blockIdx.x * blockDim.x + threadIdx.x;
        if(k >= this->O_k_length)
            return;

        #else

        for(auto k = 0u; k < this->O_k_length; k++)

        #endif
        {
            this->forces[k] = this->E_local_O_k_avg[k] - *this->E_local_avg * conj(this->O_k_avg[k]);
        }
    }

    HDINLINE
    void compute_O_k_avg_dot_x() const {
        #ifdef __CUDA_ARCH__

        const auto k = blockIdx.x * blockDim.x + threadIdx.x;
        if(k >= this->O_k_length)
            return;

        atomicAdd(this->O_k_avg_x, this->O_k_avg[k] * this->x[k]);

        #else

        for(auto k = 0u; k < this->O_k_length; k++) {
            *this->O_k_avg_x += this->O_k_avg[k] * this->x[k];
        }

        #endif
    }

    HDINLINE
    void finalize_Ax(const unsigned int num_steps) const {
        #ifdef __CUDA_ARCH__

        const auto k = blockIdx.x * blockDim.x + threadIdx.x;
        if(k >= this->O_k_length)
            return;
        {

        #else

        for(auto k = 0u; k < this->O_k_length; k++) {

        #endif

            this->Ax[k] = (
                this->Ax[k] * (1.0 / num_steps) -
                conj(this->O_k_avg[k]) * *this->O_k_avg_x
            );

        }
    }


#endif // __CUDACC__
};

} // namespace kernel


class DifferentiatePsi : public kernel::DifferentiatePsi {
private:
    const bool gpu;

    friend void evaluate_Ax(int*, std::complex<double>* x, std::complex<double>* Ax);
public:
    DifferentiatePsi(const unsigned int O_k_length, const bool gpu=true);
    ~DifferentiatePsi() noexcept(false);

    template<typename SpinEnsemble>
    void operator()(
        const Psi&                  psi,
        const Operator&             operator_,
        const SpinEnsemble&         spin_ensemble,
        std::complex<double>*       result,
        const double                rtol=1e-6
    );

    std::vector<std::complex<double>> get_O_k_avg() const;

    inline kernel::DifferentiatePsi get_kernel() const {
        return static_cast<kernel::DifferentiatePsi>(*this);
    }

#ifdef __PYTHONCC__

    template<typename SpinEnsemble>
    decltype(auto) operator()(
        const Psi&          psi,
        const Operator&     operator_,
        const SpinEnsemble& spin_ensemble,
        const double        rtol=1e-6
    ) {
        xt::pytensor<std::complex<double>, 1u> result(std::array<long int, 1u>({(long int)psi.get_num_params()}));
        (*this)(psi, operator_, spin_ensemble, result.raw_data(), rtol);

        return result;
    }

    template<typename SpinEnsemble>
    xt::pytensor<std::complex<double>, 1u> __call__(
        const Psi&          psi,
        const Operator&     operator_,
        const SpinEnsemble& spin_ensemble,
        const double        rtol=1e-6
    ) {
        return (*this)(psi, operator_, spin_ensemble, rtol);
    }

#endif // __CUDACC__

};

} // namespace rbm_on_gpu
