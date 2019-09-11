#pragma once

#include "Spins.h"
#include "cuda_complex.hpp"


namespace rbm_on_gpu {


enum class PauliMatrices : int {
    Identity = 0, SigmaX = 1, SigmaY = 2, SigmaZ = 3
};

struct MatrixElement {
    complex_t coefficient;
    Spins     spins;
};

struct MatrixElementStd {
    std::complex<float> coefficient;
    Spins                spins;
};

} // namespace rbm_on_gpu
