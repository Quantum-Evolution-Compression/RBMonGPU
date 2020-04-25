#include "RNGStates.hpp"
#include "types.h"
#include <iostream>


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


RNGStates::RNGStates(const unsigned int num_states, const bool gpu)
:
    num_states(num_states),
    rng_states_device(nullptr),
    rng_states_host(nullptr),
    gpu(gpu) {

    if(this->gpu) {
        CUDA_CHECK(cudaMalloc(&this->rng_states_device, sizeof(curandState_t) * this->num_states));

        const auto blockDim = 256u;
        kernel::initialize_random_states<<<this->num_states / blockDim + 1u, blockDim>>>(
            this->rng_states_device,
            this->num_states
        );
    }
    else {
        this->rng_states_host = new mt19937[num_states];
        for(auto i = 0u; i < this->num_states; i++) {
            this->rng_states_host[i] = mt19937(i);
        }
    }
}

RNGStates::~RNGStates() noexcept(false) {
    if(this->gpu) {
        CUDA_FREE(this->rng_states_device);
    }
    else {
        delete[] this->rng_states_host;
    }
}


}
