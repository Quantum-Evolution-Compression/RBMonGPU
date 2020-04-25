#pragma once

#ifdef __CUDACC__
    #include "curand_kernel.h"
#else
    struct curandState_t;
#endif // __CUDACC__
#include <random>


namespace rbm_on_gpu {

using namespace std;


struct RNGStates {
    bool            gpu;
    unsigned int    num_states;
    curandState_t*  rng_states_device;
    mt19937*        rng_states_host;

    RNGStates(const unsigned int num_states, const bool gpu);
    ~RNGStates() noexcept(false);
};

}  // namespace rbm_on_gpu
