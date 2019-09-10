#pragma once

#include "operator/Operator.hpp"
#include "Array.hpp"
#include "Spins.h"
#include "cuda_complex.hpp"
#include "types.h"

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pytensor.hpp"
#endif // __CUDACC__

// #include <vector>
// #include <memory>
#include <cmath>


namespace rbm_on_gpu {

namespace kernel {

class ExactSummation {
protected:

    unsigned int  num_spin_configurations;
    bool          has_total_z_symmetry;
    Spins*        allowed_spin_configurations;

public:

    inline unsigned int get_num_steps() const {
        return this->num_spin_configurations;
    }

    inline bool has_weights() const {
        return true;
    }

#ifdef __CUDACC__

    template<typename Psi_t, typename Function>
    HDINLINE
    void kernel_foreach(const Psi_t psi, Function function) const {
        #ifdef __CUDA_ARCH__
            #define SHARED __shared__
        #else
            #define SHARED
        #endif

        SHARED Spins        spins;
        SHARED complex_t    log_psi;
        SHARED double       weight;

        #ifdef __CUDA_ARCH__
        const auto spin_index = blockIdx.x;
        #else
        for(auto spin_index = 0u; spin_index < this->num_spin_configurations; spin_index++)
        #endif
        {
            if(this->has_total_z_symmetry) {
                spins = this->allowed_spin_configurations[spin_index];
            }
            else {
                spins = {(Spins::type)spin_index};
            }


            SHARED complex_t angle_ptr[Psi_t::get_max_angles()];
            psi.init_angles(angle_ptr, spins);

            #ifdef __CUDA_ARCH__
            __syncthreads();
            #endif

            psi.log_psi_s(log_psi, spins, angle_ptr);

            #ifdef __CUDA_ARCH__
            __syncthreads();
            #endif

            #ifdef __CUDA_ARCH__
            if(threadIdx.x == 0)
            #endif
            {
                weight = this->num_spin_configurations * psi.probability_s(log_psi.real());
            }

            #ifdef __CUDA_ARCH__
            __syncthreads();
            #endif

            function(spin_index, spins, log_psi, angle_ptr, weight);
        }
    }

#endif // __CUDACC__

};

} // namespace kernel

class ExactSummation : public kernel::ExactSummation {
protected:

    bool          gpu;
    unsigned int  num_spins;
    Array<Spins>* allowed_spin_configurations_vec;

public:
    ExactSummation(const unsigned int num_spins, const bool gpu);
    ~ExactSummation() noexcept(false);

    // ExactSummation copy() const {
    //     return *this;
    // }

    void set_total_z_symmetry(const int sector);

#ifdef __CUDACC__
    template<typename Psi_t, typename Function>
    inline void foreach(const Psi_t& psi, const Function& function, const int blockDim=-1) const {
        // compute list of spins and angles
        const auto psi_kernel = psi.get_kernel();
        if(psi.on_gpu()) {
            const auto blockDim_ = blockDim == -1 ? psi.get_num_angles() : blockDim;

            cuda_kernel<<<this->num_spin_configurations, blockDim_>>>(
                [=, *this] __device__ () {this->kernel_foreach(psi_kernel, function);}
            );
        }
        else {
            this->kernel_foreach(psi_kernel, function);
        }
    }
#endif

};

} // namespace rbm_on_gpu
