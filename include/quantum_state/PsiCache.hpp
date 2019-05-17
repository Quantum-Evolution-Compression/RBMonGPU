#pragma once

#include "quantum_state/psi_functions.hpp"
#include "quantum_state/PsiHistory.hpp"
#include "Spins.h"
#include "types.h"


namespace rbm_on_gpu {

#ifdef __CUDACC__

struct PsiCache {
    complex_t tanh_angles[MAX_HIDDEN_SPINS];

    PsiCache() {}

    template<typename Psi_t>
    HDINLINE PsiCache(const complex_t* angle_ptr, const Psi_t& psi) {
        this->init(angle_ptr, psi);
    }

    template<typename SpinHistory>
    HDINLINE PsiCache(const unsigned int step, const SpinHistory& spin_history) {
        this->load(step, spin_history);
    }

    HDINLINE PsiCache(const unsigned int step, const kernel::PsiHistory& psi_history) {
        this->load(step, psi_history);
    }

    template<typename Psi_t>
    HDINLINE void init(const complex_t* angle_ptr, const Psi_t& psi) {
        #ifdef __CUDA_ARCH__
            const auto j = threadIdx.x;
            if(j < psi.get_num_hidden_spins())
        #else
            for(auto j = 0u; j < psi.get_num_hidden_spins(); j++)
        #endif
        {
            this->tanh_angles[j] = my_tanh(angle_ptr[j]);
        }
    }

    template<typename SpinHistory>
    HDINLINE void load(const unsigned int step, const SpinHistory& spin_history) {
        #ifdef __CUDA_ARCH__
            const auto j = threadIdx.x;
            if(j < spin_history.num_hidden_spins)
        #else
            for(auto j = 0u; j < spin_history.num_hidden_spins; j++)
        #endif
        {
            this->tanh_angles[j] = my_tanh(spin_history.angles[step * spin_history.num_hidden_spins + j]);
        }

    }

    HDINLINE void load(const unsigned int step, const kernel::PsiHistory& psi_history) {
        #ifdef __CUDA_ARCH__
            const auto j = threadIdx.x;
            if(j < psi_history.num_hidden_spins)
        #else
            for(auto j = 0u; j < psi_history.num_hidden_spins; j++)
        #endif
        {
            this->tanh_angles[j] = psi_history.tanh_angles[step * psi_history.num_hidden_spins + j];
        }
    }

    HDINLINE void store(const unsigned int step, const kernel::PsiHistory& psi_history) const {
        #ifdef __CUDA_ARCH__
            const auto j = threadIdx.x;
            if(j < psi_history.num_hidden_spins)
        #else
            for(auto j = 0u; j < psi_history.num_hidden_spins; j++)
        #endif
        {
            psi_history.tanh_angles[step * psi_history.num_hidden_spins + j] = this->tanh_angles[j];
        }
    }
};

#endif

} // namespace rbm_on_gpu
