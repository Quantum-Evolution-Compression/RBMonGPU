#pragma once

#include "operator/Operator.hpp"
#include "Array.hpp"
#include "Spins.h"
#include "cuda_complex.hpp"
#include "types.h"

#include <memory>
#include <cmath>


namespace rbm_on_gpu {

namespace kernel {

struct SpecialExactSummation {

    unsigned int  num_spin_configurations;


    inline unsigned int get_num_steps() const {
        return this->num_spin_configurations;
    }

    inline bool has_weights() const {
        return true;
    }

#ifdef __CUDACC__

    template<typename Psi_t, typename Function>
    HDINLINE
    void kernel_foreach(Psi_t psi, Function function) const {
        #include "cuda_kernel_defines.h"

        SHARED Spins        spins[2];
        SHARED complex_t    log_psi[2];
        SHARED double       weight;
        const auto          N = psi.get_num_spins();

        #ifdef __CUDA_ARCH__
        const auto double_spin_index = blockIdx.x;
        #else
        for(auto double_spin_index = 0u; double_spin_index < this->num_spin_configurations; double_spin_index++)
        #endif
        {
            spins[0] = Spins(double_spin_index, N);
            spins[1] = Spins(double_spin_index >> N, N);

            SHARED typename Psi_t::Angles angles;

            psi.log_psi_s(log_psi[0], spins[0], angles);
            psi.log_psi_s(log_psi[1], spins[1], angles);

            SYNC;

            SINGLE {
                weight = this->num_spin_configurations * (
                    psi.probability_s(log_psi[0].real()) *
                    psi.probability_s(log_psi[1].real()) / double(
                        1u << spins[0].extract_first_n(N / 2).hamming_distance(spins[1].extract_first_n(N / 2))
                    )
                );
            }

            SYNC;

            function(double_spin_index, spins, log_psi, angles, weight);
        }
    }

#endif // __CUDACC__

    SpecialExactSummation get_kernel() const {
        return *this;
    }

};

} // namespace kernel

struct SpecialExactSummation : public kernel::SpecialExactSummation {

    bool          gpu;
    unsigned int  num_spins;

    SpecialExactSummation(const unsigned int num_spins, const bool gpu);

#ifdef __CUDACC__
    template<typename Psi_t, typename Function>
    inline void foreach(const Psi_t& psi, const Function& function, const int blockDim=-1) const {
        auto this_kernel = this->get_kernel();
        const auto psi_kernel = psi.get_kernel();
        if(psi.gpu) {
            const auto blockDim_ = blockDim == -1 ? psi.get_width() : blockDim;

            cuda_kernel<<<this->num_spin_configurations, blockDim_>>>(
                [=] __device__ () {this_kernel.kernel_foreach(psi_kernel, function);}
            );
        }
        else {
            this_kernel.kernel_foreach(psi_kernel, function);
        }
    }
#endif

};

} // namespace rbm_on_gpu
