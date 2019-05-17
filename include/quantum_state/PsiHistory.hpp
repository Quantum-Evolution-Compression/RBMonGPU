#pragma once

#include "types.h"

namespace rbm_on_gpu {

namespace kernel {

struct PsiHistory {

    unsigned int  num_hidden_spins;
    complex_t*    tanh_angles;

    inline PsiHistory get_kernel() const {
        return *this;
    }
};

} // namespace kernel

struct PsiHistory : public kernel::PsiHistory {

    const bool gpu;

    PsiHistory(const unsigned int num_steps, const unsigned int num_hidden_spins, const bool gpu);
    ~PsiHistory() noexcept(false);
};

} // namespace rbm_on_gpu
