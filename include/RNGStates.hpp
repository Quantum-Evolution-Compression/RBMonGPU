#pragma once


namespace rbm_on_gpu {

struct RNGStates {
    bool         gpu;
    unsigned int num_states;
    void*        rng_states;

    RNGStates(const unsigned int num_states, const bool gpu);
    ~RNGStates() noexcept(false);
};

}  // namespace rbm_on_gpu
