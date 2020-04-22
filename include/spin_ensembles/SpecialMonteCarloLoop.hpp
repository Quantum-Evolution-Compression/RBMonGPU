#pragma once

#include "operator/Operator.hpp"
#include "Spins.h"
#include "Array.hpp"
#include "random.h"
#include "cuda_complex.hpp"
#include "types.h"

#ifdef __CUDACC__
    #include "curand_kernel.h"
#else
    struct curandState_t;
#endif // __CUDACC__

#include <vector>
#include <memory>
#include <random>


namespace rbm_on_gpu {

namespace kernel {

struct SpecialMonteCarloLoop {

    curandState_t*  random_states;
    std::mt19937*   random_state_host;
    unsigned int    num_samples;
    unsigned int    num_sweeps;
    unsigned int    num_thermalization_sweeps;
    unsigned int    num_markov_chains;

    unsigned int    num_mc_steps_per_chain;

    unsigned int*   acceptances;
    unsigned int*   rejections;


    inline unsigned int get_num_steps() const {
        return this->num_samples;
    }

    inline bool has_weights() const {
        return false;
    }

#ifdef __CUDACC__

    template<typename Psi_t, typename Function>
    HDINLINE
    void kernel_foreach(const Psi_t psi, Function function) const {
        // ##################################################################################
        //
        // Call with gridDim.x = number of markov chains, blockDim.x = number of hidden spins
        //
        // ##################################################################################
        #include "cuda_kernel_defines.h"

        SHARED unsigned int markov_index;
        #ifdef __CUDA_ARCH__
            __shared__ curandState_t local_random_state;
            if(threadIdx.x == 0) {
                markov_index = blockIdx.x;
                local_random_state = this->random_states[blockIdx.x];
            }
        #else
            markov_index = 0u;
            std::mt19937 local_random_state = this->random_state_host[markov_index];
        #endif

        SHARED Spins spins[2];

        SINGLE {
            spins[0] = Spins::random(&local_random_state, psi.get_num_spins());
            spins[1] = Spins::random(&local_random_state, psi.get_num_spins());
        }
        SYNC;

        SHARED typename Psi_t::Angles angles;

        SHARED complex_t log_psi[2];
        SHARED double log_psi_real[2];

        psi.log_psi_s_real(log_psi_real[0], spins[0], angles);
        psi.log_psi_s_real(log_psi_real[1], spins[1], angles);

        this->thermalize(psi, log_psi_real, spins, angles, &local_random_state);

        SHARED_MEM_LOOP_BEGIN(mc_step_within_chain, this->num_mc_steps_per_chain) {

            SHARED_MEM_LOOP_BEGIN(
                i,
                this->num_sweeps * psi.get_num_spins()
            ) {
                this->mc_update(psi, log_psi_real, spins, angles, &local_random_state);

                SHARED_MEM_LOOP_END(i);
            }

            psi.log_psi_s(log_psi[0], spins[0], angles);
            psi.log_psi_s(log_psi[1], spins[1], angles);

            function(
                mc_step_within_chain * this->num_markov_chains + markov_index,
                spins,
                log_psi,
                angles,
                1.0
            );

            SHARED_MEM_LOOP_END(mc_step_within_chain);
        }

        #ifdef __CUDA_ARCH__
        if(threadIdx.x == 0) {
            this->random_states[markov_index] = local_random_state;
        }
        #else
        this->random_state_host[markov_index] = local_random_state;
        #endif
    }

    template<typename Psi_t>
    HDINLINE
    void thermalize(
        const Psi_t& psi,
        double* log_psi_real,
        Spins* spins,
        typename Psi_t::Angles& angles,
        void* local_random_state
    ) const {
        #include "cuda_kernel_defines.h"

        SHARED_MEM_LOOP_BEGIN(i, this->num_thermalization_sweeps * psi.get_num_spins()) {
            this->mc_update(psi, log_psi_real, spins, angles, local_random_state);

            SHARED_MEM_LOOP_END(i);
        }
    }

    template<typename Psi_t>
    HDINLINE
    void mc_update(
        const Psi_t& psi,
        double* log_psi_real,
        Spins* spins,
        typename Psi_t::Angles& angles,
        void* local_random_state
    ) const {
        #include "cuda_kernel_defines.h"

        SHARED unsigned int position;
        SHARED unsigned int ab;
        SHARED double hamming_factor;

        SINGLE {
            const auto global_position = random_uint64(local_random_state) % (2u * psi.get_num_spins());
            position = global_position % psi.get_num_spins();
            ab = global_position / psi.get_num_spins();

            spins[ab] = spins[ab].flip(position);
            if(position < psi.get_num_spins() / 2) {
                hamming_factor = spins[0].bit_at(position) == spins[1].bit_at(position) ? 2.0 : 0.5;
            }
            else {
                hamming_factor = 1.0;
            }
        }
        SYNC;

        SHARED double next_log_psi_real;
        psi.log_psi_s_real(next_log_psi_real, spins[ab], angles);

        SHARED double ratio;
        SINGLE {
            ratio = exp(
                2.0 * (
                    next_log_psi_real -
                    log_psi_real[ab]
                )
            ); * hamming_factor;
            // ratio = 2.0;

            if(ratio > 1.0 || random_real(local_random_state) <= ratio) {
                log_psi_real[ab] = next_log_psi_real;
                generic_atomicAdd(this->acceptances, 1u);
            }
            else {
                spins[ab] = spins[ab].flip(position);
                generic_atomicAdd(this->rejections, 1u);
            }
        }
        SYNC;
    }

#endif // __CUDACC__

};

} // namespace kernel


struct SpecialMonteCarloLoop : public kernel::SpecialMonteCarloLoop {
    bool gpu;

    Array<unsigned int> acceptances_ar;
    Array<unsigned int> rejections_ar;

    void allocate_memory();

    SpecialMonteCarloLoop(
        const unsigned int num_samples,
        const unsigned int num_sweeps,
        const unsigned int num_thermalization_sweeps,
        const unsigned int num_markov_chains,
        const bool         gpu
    );
    SpecialMonteCarloLoop(SpecialMonteCarloLoop& other);
    ~SpecialMonteCarloLoop() noexcept(false);

#ifdef __CUDACC__
    template<typename Psi_t, typename Function>
    inline void foreach(const Psi_t& psi, const Function& function, const int blockDim=-1) {
        auto this_kernel = this->get_kernel();
        auto psi_kernel = psi.get_kernel();

        this->acceptances_ar.clear();
        this->rejections_ar.clear();

        #ifdef TIMING
            const auto begin = clock::now();
        #endif

        if(this->gpu) {
            const auto blockDim_ = blockDim == -1 ? psi.get_width() : blockDim;

            cuda_kernel<<<this->num_markov_chains, blockDim_>>>(
                [=] __device__ () {this_kernel.kernel_foreach(psi_kernel, function);}
            );
        }
        else {
            this_kernel.kernel_foreach(psi_kernel, function);
        }

        this->acceptances_ar.update_host();
        this->rejections_ar.update_host();

        #ifdef TIMING
            if(this->gpu) {
                cudaDeviceSynchronize();
            }
            const auto end = clock::now();
            log_duration("SpecialMonteCarloLoop::foreach", end - begin);
        #endif
    }
#endif

    inline kernel::SpecialMonteCarloLoop get_kernel() const {
        return static_cast<kernel::SpecialMonteCarloLoop>(*this);
    }
};


} // namespace rbm_on_gpu
