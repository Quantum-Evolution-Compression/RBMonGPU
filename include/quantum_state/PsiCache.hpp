#pragma once

#include "quantum_state/psi_functions.hpp"
#include "Spins.h"
#include "types.h"


namespace rbm_on_gpu {

// #ifdef __CUDACC__

struct PsiAngles {
    complex_t angles[MAX_HIDDEN_SPINS];

    PsiAngles() = default;

    template<typename Psi_t>
    HDINLINE void init(const Psi_t& psi, const PsiAngles& other) {
        #include "cuda_kernel_defines.h"

        MULTI(j, psi.get_num_hidden_spins())
        {
            this->angles[j] = other[j];
        }
    }

    template<typename Psi_t>
    HDINLINE void init(const Psi_t& psi, const Spins& spins) {
        #include "cuda_kernel_defines.h"

        MULTI(j, psi.get_num_hidden_spins())
        {
            this->angles[j] = psi.angle(j, spins);
        }
    }

    HDINLINE complex_t& operator[](const unsigned int i) {
        return this->angles[i];
    }

    HDINLINE complex_t& operator[](const int i) {
        return this->angles[i];
    }

    HDINLINE const complex_t& operator[](const unsigned int i) const {
        return this->angles[i];
    }

    HDINLINE const complex_t& operator[](const int i) const {
        return this->angles[i];
    }
};

struct PsiDerivatives {
    complex_t tanh_angles[MAX_HIDDEN_SPINS];

    template<typename Psi_t>
    HDINLINE void init(const Psi_t& psi, const PsiAngles& psi_angles) {
        #include "cuda_kernel_defines.h"

        MULTI(j, psi.get_num_hidden_spins())
        {
            this->tanh_angles[j] = my_tanh(psi_angles.angles[j]);
        }
    }
};

// #endif

} // namespace rbm_on_gpu
