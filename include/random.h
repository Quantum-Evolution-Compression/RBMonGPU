#pragma once

#include "types.h"
#include <builtin_types.h>
#include <cstdint>
#include <random>
#include <bitset>

#ifdef __CUDACC__
    #include "curand_kernel.h"
#else
    struct curandState_t;
#endif


namespace rbm_on_gpu {

using namespace std;


HDINLINE uint64_t random_uint64(void* rng_state) {
    #ifdef __CUDA_ARCH__
        return curand(reinterpret_cast<curandState_t*>(rng_state));
    #else
        std::uniform_int_distribution<uint64_t> random_spin_conf(0, UINT64_MAX);
        return random_spin_conf(*reinterpret_cast<std::mt19937*>(rng_state));
    #endif
}


HDINLINE double random_real(void* rng_state) {
    #ifdef __CUDA_ARCH__
        return curand_uniform(reinterpret_cast<curandState_t*>(rng_state));
    #else
        std::uniform_real_distribution<double> uniform_real(0.0, 1.0);
        return uniform_real(*reinterpret_cast<std::mt19937*>(rng_state));
    #endif
}


HDINLINE unsigned int count_bits(const uint64_t number) {
    #ifdef __CUDA_ARCH__
        return __popcll(number);
    #else
        return __builtin_popcountll(number);
    #endif
}


HDINLINE uint64_t random_n_over_k_bitstring(unsigned int n, unsigned int k, void* rng_state) {
    uint64_t result = random_uint64(rng_state) & ((1 << n) - 1);
    int diff = (int)k - (int)count_bits(result);

    while(diff != 0) {
        unsigned int pos = random_uint64(rng_state) % n;
        while(static_cast<bool>(result & (1 << pos)) == static_cast<bool>(diff > 0)) {
            pos = (pos + 1) % n;
        }
        result ^= 1 << pos;

        if(diff > 0) {
            diff--;
        } else {
            diff++;
        }
    }

    return result;
}

} // rbm_on_gpu
