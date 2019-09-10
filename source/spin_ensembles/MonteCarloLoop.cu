#include "spin_ensembles/MonteCarloLoop.hpp"
#include "quantum_state/Psi.hpp"

#include <cassert>


namespace rbm_on_gpu {

__global__ void kernel_initialize_random_states(curandState_t* random_states, const unsigned int num_markov_chains) {
    const auto markov_index = blockIdx.x * blockDim.x + threadIdx.x;

    if(markov_index < num_markov_chains) {
        curand_init(0, markov_index, 0u, &random_states[markov_index]);
    }
}


MonteCarloLoop::MonteCarloLoop(
    const unsigned int num_samples,
    const unsigned int num_sweeps,
    const unsigned int num_thermalization_sweeps,
    const unsigned int num_markov_chains,
    const bool         gpu
) : gpu(gpu) {
    this->num_samples = num_samples;
    this->num_sweeps = num_sweeps;
    this->num_thermalization_sweeps = num_thermalization_sweeps;
    this->num_markov_chains = num_markov_chains;
    this->has_total_z_symmetry = false;

    this->allocate_memory();
}

MonteCarloLoop::MonteCarloLoop(const MonteCarloLoop& other) : gpu(other.gpu) {
    this->num_samples = other.num_samples;
    this->num_sweeps = other.num_sweeps;
    this->num_thermalization_sweeps = other.num_thermalization_sweeps;
    this->num_markov_chains = other.num_markov_chains;
    this->has_total_z_symmetry = other.has_total_z_symmetry;
    this->symmetry_sector = other.symmetry_sector;

    this->allocate_memory();
}

MonteCarloLoop::~MonteCarloLoop() noexcept(false) {
    if(this->gpu) {
        CUDA_FREE(this->random_states)
    }
    else {
        delete[] this->random_state_host;
    }
}

void MonteCarloLoop::allocate_memory() {
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
