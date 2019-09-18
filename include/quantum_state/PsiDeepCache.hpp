#pragma once

#include "Spins.h"
#include "types.h"


namespace rbm_on_gpu {

// #ifdef __CUDACC__

struct PsiDeepAngles {
    static constexpr unsigned int max_width = MAX_SPINS;

    complex_t activations[max_width];

    PsiDeepAngles() = default;

    template<typename Psi_t>
    HDINLINE void init(const Psi_t& psi, const PsiDeepAngles& other) {
    }

    template<typename Psi_t>
    HDINLINE void init(const Psi_t& psi, const Spins& spins) {
    }

};

// #endif

} // namespace rbm_on_gpu
