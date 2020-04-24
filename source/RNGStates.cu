#include "RNGStates.hpp"
#ifdef __CUDACC__
    #include "curand_kernel.h"
#else
    struct curandState_t;
#endif // __CUDACC__
#include <random>
#include "types.h"


using namespace std;


namespace rbm_on_gpu {

namespace kernel {

__global__ void initialize_random_states(curandState_t* random_states, const unsigned int num_states) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < num_states) {
        curand_init(0, idx, 0u, &random_states[idx]);
    }
}

}


RNGStates::RNGStates(const unsigned int num_states, const bool gpu) : num_states(num_states), gpu(gpu) {
    if(this->gpu) {
        CUDA_CHECK(cudaMalloc(&this->rng_states, sizeof(curandState_t) * this->num_states));

        const auto blockDim = 256u;
        kernel::initialize_random_states<<<this->num_states / blockDim + 1u, blockDim>>>(
            reinterpret_cast<curandState_t*>(this->rng_states),
            this->num_states
        );
    }
    else {
        this->rng_states = new mt19937[num_states];
        for(auto i = 0u; i < this->num_states; i++) {
            reinterpret_cast<mt19937*>(this->rng_states)[i] = mt19937(i);
        }
    }
}

RNGStates::~RNGStates() noexcept(false) {
    if(this->gpu) {
        CUDA_FREE(this->rng_states);
    }
    else {
        delete[] reinterpret_cast<mt19937*>(this->rng_states);
    }
}


}
