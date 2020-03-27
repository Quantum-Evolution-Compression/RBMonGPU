#pragma once

#include "operator/Operator.hpp"
#include "Spins.h"
#include "random.h"
#include "cuda_complex.hpp"
#include "types.h"

#ifdef __CUDACC__
    #include "curand_kernel.h"
#else
    struct curandState_t;
#endif // __CUDACC__

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pytensor.hpp"
#endif // __PYTHONCC__

#include <vector>
#include <memory>
#include <random>


namespace rbm_on_gpu {

namespace kernel {

class MonteCarloLoop {
public:
    curandState_t*  random_states;
    std::mt19937*   random_state_host;
    unsigned int    num_samples;
    unsigned int    num_sweeps;
    unsigned int    num_thermalization_sweeps;
    unsigned int    num_markov_chains;

    bool            has_total_z_symmetry;
    int             symmetry_sector;

    unsigned int    num_mc_steps_per_chain;

public:
    inline unsigned int get_num_steps() const {
        return this->num_samples;
    }

    inline bool has_weights() const {
        return false;
    }

#ifdef __CUDACC__

    template<bool total_z_symmetry, typename Psi_t, typename Function>
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

        SHARED Spins spins;

        SINGLE {
            if(total_z_symmetry) {
                spins = Spins(
                    random_n_over_k_bitstring(
                        psi.get_num_spins(),
                        (this->symmetry_sector + psi.get_num_spins()) / 2,
                        &local_random_state
                    ),
                    psi.get_num_spins()
                );
            }
            else {
                spins = Spins::random(&local_random_state, psi.get_num_spins());
            }
        }
        SYNC;

        SHARED typename Psi_t::Angles angles;
        angles.init(psi, spins);
        SYNC;

        SHARED complex_t log_psi;
        SHARED double log_psi_real;

        psi.log_psi_s_real(log_psi_real, spins, angles);

        this->thermalize<total_z_symmetry>(psi, log_psi_real, spins, &local_random_state, angles);

        SHARED_MEM_LOOP_BEGIN(mc_step_within_chain, this->num_mc_steps_per_chain) {
        // for(auto mc_step_within_chain = 0u; mc_step_within_chain < this->num_mc_steps_per_chain; mc_step_within_chain++) {

            SHARED_MEM_LOOP_BEGIN(i, this->num_sweeps * psi.get_num_spins()) {
            // for(auto i = 0u; i < this->num_sweeps * psi.get_num_spins(); i++) {
                this->mc_update<total_z_symmetry>(psi, log_psi_real, spins, &local_random_state, angles);

                SHARED_MEM_LOOP_END(i);
            }

            psi.log_psi_s(log_psi, spins, angles);

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

    template<bool total_z_symmetry, typename Psi_t>
    HDINLINE
    void thermalize(const Psi_t& psi, double& log_psi_real, Spins& spins, void* local_random_state, typename Psi_t::Angles& angles) const {
        #include "cuda_kernel_defines.h"

        SHARED_MEM_LOOP_BEGIN(i, this->num_thermalization_sweeps * psi.get_num_spins()) {
        // for(auto i = 0u; i < this->num_thermalization_sweeps * psi.get_num_spins(); i++) {
            this->mc_update<total_z_symmetry>(psi, log_psi_real, spins, local_random_state, angles);

            SHARED_MEM_LOOP_END(i);
        }
    }

    template<bool total_z_symmetry, typename Psi_t>
    HDINLINE
    void mc_update(const Psi_t& psi, double& log_psi_real, Spins& spins, void* local_random_state, typename Psi_t::Angles& angles) const {
        #include "cuda_kernel_defines.h"

        SHARED int position;
        SHARED int second_position;

        SINGLE {
            position = random_uint64(local_random_state) % psi.get_num_spins();
            spins = spins.flip(position);
        }
        SYNC;

        MULTI(j, psi.get_num_angles()) {
            psi.flip_spin_of_jth_angle(j, position, spins, angles);
        }

        if(total_z_symmetry) {
            SYNC;

            SINGLE {
                while(true) {
                    second_position = random_uint64(local_random_state) % psi.get_num_spins();
                    if(spins[second_position] == spins[position]) {
                        spins = spins.flip(second_position);
                        break;
                    }
                }
            }
            SYNC;
            MULTI(j, psi.get_num_angles()) {
                psi.flip_spin_of_jth_angle(j, second_position, spins, angles);
            }
        }

        SHARED double next_log_psi_real;
        psi.log_psi_s_real(next_log_psi_real, spins, angles);

        SHARED bool spin_flip;
        SHARED double ratio;
        SINGLE {
            ratio = exp(2.0 * (next_log_psi_real - log_psi_real));

            if(ratio > 1.0 || random_real(local_random_state) <= ratio) {
                log_psi_real = next_log_psi_real;
                spin_flip = true;
            }
            else {
                spin_flip = false;
            }
        }
        SYNC;

        if(!spin_flip) {
            // flip back spin(s)

            SINGLE {
                spins = spins.flip(position);
            }
            SYNC;
            MULTI(j, psi.get_num_angles()) {
                psi.flip_spin_of_jth_angle(j, position, spins, angles);
            }

            if(total_z_symmetry) {
                SINGLE {
                    spins = spins.flip(second_position);
                }
                SYNC;
                MULTI(j, psi.get_num_angles()) {
                    psi.flip_spin_of_jth_angle(j, second_position, spins, angles);
                }
            }
        }
    }

#endif // __CUDACC__

};

} // namespace kernel


class MonteCarloLoop : public kernel::MonteCarloLoop {
private:
    bool gpu;

    void allocate_memory();
public:
    MonteCarloLoop(
        const unsigned int num_samples,
        const unsigned int num_sweeps,
        const unsigned int num_thermalization_sweeps,
        const unsigned int num_markov_chains,
        const bool         gpu
    );
    MonteCarloLoop(const MonteCarloLoop& other);
    ~MonteCarloLoop() noexcept(false);

    inline void set_total_z_symmetry(const int sector) {
        this->symmetry_sector = sector;
        this->has_total_z_symmetry = true;
    }

#ifdef __CUDACC__
    template<typename Psi_t, typename Function>
    inline void foreach(const Psi_t& psi, const Function& function, const int blockDim=-1) const {
        auto this_kernel = this->get_kernel();
        auto psi_kernel = psi.get_kernel();

        #ifdef TIMING
            const auto begin = clock::now();
        #endif

        if(this->gpu) {
            const auto blockDim_ = blockDim == -1 ? psi.get_width() : blockDim;

            if(this->has_total_z_symmetry) {
                cuda_kernel<<<this->num_markov_chains, blockDim_>>>(
                    [=] __device__ () {this_kernel.kernel_foreach<true>(psi_kernel, function);}
                );
            }
            else {
                cuda_kernel<<<this->num_markov_chains, blockDim_>>>(
                    [=] __device__ () {this_kernel.kernel_foreach<false>(psi_kernel, function);}
                );
            }
        }
        else {
            if(this->has_total_z_symmetry) {
                this_kernel.kernel_foreach<true>(psi_kernel, function);
            }
            else {
                this_kernel.kernel_foreach<false>(psi_kernel, function);
            }
        }

        #ifdef TIMING
            if(this->gpu) {
                cudaDeviceSynchronize();
            }
            const auto end = clock::now();
            log_duration("MonteCarloLoop::foreach", end - begin);
        #endif
    }
#endif

    inline kernel::MonteCarloLoop get_kernel() const {
        return static_cast<kernel::MonteCarloLoop>(*this);
    }
};


} // namespace rbm_on_gpu
