#pragma once

#include "quantum_state/psi_functions.hpp"
#include "Spins.h"
#include "types.h"


namespace rbm_on_gpu {

#ifdef __CUDACC__

struct PsiW3Cache {
    complex_t   Z_f[MAX_F];
    complex_t*  f_angles;
    complex_t   tanh_j_angles[MAX_HIDDEN_SPINS];

    PsiW3Cache() {}

    template<typename PsiW3_t>
    HDINLINE PsiW3Cache(complex_t* angle_ptr, const PsiW3_t& psi) {
        this->init(angle_ptr, psi);
    }

    template<typename PsiW3_t>
    HDINLINE void init(complex_t* angle_ptr, const PsiW3_t& psi) {
        #ifdef __CUDA_ARCH__
            const auto j = threadIdx.x;
            if(j < psi.get_num_hidden_spins())
        #else
            for(auto j = 0u; j < psi.get_num_hidden_spins(); j++)
        #endif
        {
            this->tanh_j_angles[j] = my_tanh(angle_ptr[j]);
        }

        #ifdef __CUDA_ARCH__
        __syncthreads();
        #endif

        this->f_angles = angle_ptr + psi.get_num_hidden_spins();

        #ifdef __CUDA_ARCH__
            const auto f = threadIdx.x;
            if(f < psi.F)
        #else
            for(auto f = 0u; f < psi.F; f++)
        #endif
        {
            this->Z_f[f] = complex_t(0.0, 0.0);
            for(auto j = 0u; j < psi.M; j++) {
                this->Z_f[f] += this->tanh_j_angles[j] * psi.Y[f * psi.M + j] * 2.0 * this->f_angles[f];
            }
        }
    }
};

#endif

} // namespace rbm_on_gpu
