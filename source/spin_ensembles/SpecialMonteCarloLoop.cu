#ifdef ENABLE_SPECIAL_MONTE_CARLO

#include "spin_ensembles/SpecialMonteCarloLoop.hpp"
#include "quantum_state/Psi.hpp"

#include <cassert>


namespace rbm_on_gpu {

__global__ void kernel_initialize_random_states(curandState_t* random_states, const unsigned int num_markov_chains) {
    const auto markov_index = blockIdx.x * blockDim.x + threadIdx.x;

    if(markov_index < num_markov_chains) {
        curand_init(0, markov_index, 0u, &random_states[markov_index]);
    }
}


SpecialMonteCarloLoop::SpecialMonteCarloLoop(
    const unsigned int num_samples,
    const unsigned int num_sweeps,
    const unsigned int num_thermalization_sweeps,
    const unsigned int num_markov_chains,
    const bool         gpu
) : acceptances_ar(1, gpu), rejections_ar(1, gpu), gpu(gpu) {
    this->num_samples = num_samples;
    this->num_sweeps = num_sweeps;
    this->num_thermalization_sweeps = num_thermalization_sweeps;
    this->num_markov_chains = num_markov_chains;

    this->num_mc_steps_per_chain = this->num_samples / this->num_markov_chains;

    this->acceptances = this->acceptances_ar.data();
    this->rejections = this->rejections_ar.data();

    this->allocate_memory();
}

SpecialMonteCarloLoop::SpecialMonteCarloLoop(SpecialMonteCarloLoop& other)
    :
    acceptances_ar(1, other.gpu),
    rejections_ar(1, other.gpu),
    gpu(other.gpu)
{
    this->num_samples = other.num_samples;
    this->num_sweeps = other.num_sweeps;
    this->num_thermalization_sweeps = other.num_thermalization_sweeps;
    this->num_markov_chains = other.num_markov_chains;

    this->num_mc_steps_per_chain = this->num_samples / this->num_markov_chains;

    this->acceptances = this->acceptances_ar.data();
    this->rejections = this->rejections_ar.data();

    this->allocate_memory();
}

SpecialMonteCarloLoop::~SpecialMonteCarloLoop() noexcept(false) {
    if(this->gpu) {
        CUDA_FREE(this->random_states)
    }
    else {
        delete[] this->random_state_host;
    }
}

void SpecialMonteCarloLoop::allocate_memory() {
    assert(this->num_samples % this->num_markov_chains == 0u);

    if(this->gpu) {
        CUDA_CHECK(cudaMalloc(&this->random_states, sizeof(curandState_t) * this->num_markov_chains))

        const auto blockDim = 256u;
        kernel_initialize_random_states<<<this->num_markov_chains / blockDim + 1u, blockDim>>>(this->random_states, this->num_markov_chains);
    }
    else {
        assert(this->num_markov_chains == 1u);

        this->random_state_host = new std::mt19937[this->num_markov_chains];
        for(auto i = 0u; i < this->num_markov_chains; i++) {
            this->random_state_host[i] = std::mt19937(i);
        }
    }
}

} // namespace rbm_on_gpu


#endif  // ENABLE_SPECIAL_MONTE_CARLO
