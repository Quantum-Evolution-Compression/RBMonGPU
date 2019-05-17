#pragma once

#include "Spins.h"
#include "cuda_complex.hpp"
#include "types.h"

#include <cassert>
#include <complex>
#include <vector>


namespace rbm_on_gpu {

namespace kernel {

struct SpinHistory {
    unsigned int  num_steps;
    unsigned int  num_angles;

    Spins*      spins;
    complex_t*  log_psi;
    complex_t*  angles;
    double*     weights;

    bool        enable_angles;

    inline unsigned int get_num_steps() const {
        return this->num_steps;
    }

    inline bool has_weights() const {
        return true;
    }

#ifdef __CUDACC__

    template<typename Function>
    HDINLINE
    void kernel_foreach(Function function) const {
        #ifdef __CUDA_ARCH__

        #define SHARED __shared__
        const auto step = blockIdx.x;

        #else

        #define SHARED
        for(auto step = 0u; step < this->num_steps; step++)

        #endif
        {
            SHARED Spins        spins;
            SHARED complex_t    log_psi;
            SHARED double       weight;

            #ifdef __CUDA_ARCH__

            if(threadIdx.x == 0) {
                spins = this->spins[step];
                log_psi = this->log_psi[step];
                weight = (this->weights != nullptr) ? this->weights[step] : 1.0;
            }
            __shared__ complex_t angle_ptr[MAX_ANGLES];

            if(this->enable_angles) {
                if(threadIdx.x < this->num_angles) {
                    angle_ptr[threadIdx.x] = this->angles[step * this->num_angles + threadIdx.x];
                }
            }
            __syncthreads();

            #else

            spins = this->spins[step];
            log_psi = this->log_psi[step];
            complex_t angle_ptr[MAX_ANGLES];
            if(this->enable_angles) {
                for(auto j = 0u; j < this->num_angles; j++) {
                    angle_ptr[j] = this->angles[step * this->num_angles + j];
                }
            }
            weight = (this->weights != nullptr) ? this->weights[step] : 1.0;

            #endif

            function(step, spins, log_psi, angle_ptr, weight);
        }
    }

#endif

};

} // namespace kernel


struct SpinHistory : public kernel::SpinHistory {
    const bool gpu;

    SpinHistory(const unsigned int num_steps, const unsigned int num_angles, const bool enable_weights, const bool gpu);
    ~SpinHistory() noexcept(false);

    inline kernel::SpinHistory get_kernel() const {
        return static_cast<kernel::SpinHistory>(*this);
    }

    template<typename Psi_t, typename Generator>
    void fill(const Psi_t& psi, const Generator& generator) const;

    decltype(auto) get_spins() const {
        return std::vector<Spins>(this->spins, this->spins + this->num_steps);
    }
    decltype(auto) get_angles() const {
        using complex = std::complex<double>;
        return std::vector<complex>((complex*)this->angles, (complex*)this->angles + this->num_steps);
    }

    inline void toggle_angles(const bool enable_angles) {
        this->enable_angles = enable_angles;
    }

#ifdef __CUDACC__
    template<typename Psi_t, typename Function>
    inline void foreach(const Psi_t&, Function function, const int blockDim=-1) const {
        if(this->gpu) {
            const auto this_kernel = this->get_kernel();

            const auto blockDim_ = blockDim == -1 ? this->num_angles : blockDim;

            cuda_kernel<<<this->num_steps, blockDim_>>>(
                [=] __device__ () {this_kernel.kernel_foreach(function);}
            );
        }
        else {
            this->kernel_foreach(function);
        }
    }
#endif
};

} // namespace RBMonGPU
