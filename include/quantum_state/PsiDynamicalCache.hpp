#pragma once

#include "quantum_state/psi_functions.hpp"
#include "Spins.h"
#include "types.h"


namespace rbm_on_gpu {

#ifdef __CUDACC__

struct PsiDynamicalCache {
    complex_t tanh_angles[MAX_HIDDEN_SPINS];
    const complex_t* angles;

    PsiDynamicalCache() {}

    template<typename Psi_t>
    HDINLINE PsiDynamicalCache(const complex_t* angle_ptr, const Psi_t& psi) {
        this->init(angle_ptr, psi);
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
            auto angle = angle_ptr[j];
            if(psi.hidden_spin_type_list[j] == 1) {
                angle = angle * angle;
            }
            this->tanh_angles[j] = my_tanh(angle);
        }

        #ifdef __CUDA_ARCH__
        if(threadIdx.x == 0)
        #endif
        this->angles = angle_ptr;
    }
};

#endif

} // namespace rbm_on_gpu
