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

        #ifdef __CUDA_ARCH__

            const auto markov_index = blockIdx.x;

            __shared__ curandState_t local_random_state;
            __shared__ Spins spins;

            if(threadIdx.x == 0) {
                local_random_state = this->random_states[markov_index];
                if(total_z_symmetry) {
                    spins = Spins({
                        random_n_over_k_bitstring(
                            psi.get_num_spins(),
                            (this->symmetry_sector + psi.get_num_spins()) / 2,
                            &local_random_state
                        )
                    });
                }
                else {
                    spins = Spins::random(&local_random_state);
                }
            }
            __syncthreads();

            __shared__ complex_t angle_ptr[Psi_t::get_max_angles()];
            psi.init_angles(angle_ptr, spins);

            __syncthreads();

            #define SHARED __shared__

        #else

            const auto markov_index = 0;

            std::mt19937 local_random_state = this->random_state_host[markov_index];
            Spins spins;
            if(total_z_symmetry) {
                spins = Spins({
                    random_n_over_k_bitstring(
                        psi.get_num_spins(),
                        (this->symmetry_sector + psi.get_num_spins()) / 2,
                        &local_random_state
                    )
                });
            }
            else {
                spins = Spins::random(&local_random_state);
            }

            complex_t angle_ptr[Psi_t::get_max_angles()];
            psi.init_angles(angle_ptr, spins);

            #define SHARED

        #endif

        this->thermalize<total_z_symmetry>(psi, spins, this->num_thermalization_sweeps, &local_random_state, angle_ptr);

        SHARED complex_t log_psi;
        // This need not to be shared. It's just a question of speed.
        SHARED double log_psi_real;

        psi.log_psi_s_real(log_psi_real, spins, angle_ptr);

        const auto num_mc_steps_per_chain = this->num_samples / this->num_markov_chains;

        for(auto mc_step_within_chain = 0u; mc_step_within_chain < num_mc_steps_per_chain; mc_step_within_chain++) {

            for(auto i = 0u; i < this->num_sweeps * psi.get_num_spins(); i++) {
                this->mc_update<total_z_symmetry>(psi, spins, log_psi_real, &local_random_state, angle_ptr);
            }

            psi.log_psi_s(log_psi, spins, angle_ptr);

            #ifdef __CUDA_ARCH__
            const auto mc_step = mc_step_within_chain * this->num_markov_chains + blockIdx.x;
            #else
            const auto mc_step = mc_step_within_chain;
            #endif

            function(mc_step, spins, log_psi, angle_ptr, 1.0);
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
    void thermalize(const Psi_t& psi, Spins& spins, const unsigned int num_sweeps, void* local_random_state, complex_t* angle_ptr) const {
        #ifdef __CUDA_ARCH__
        __shared__ double log_psi_real;
        #else
        double log_psi_real;
        #endif

        psi.log_psi_s_real(log_psi_real, spins, angle_ptr);

        for(auto i = 0u; i < num_sweeps * psi.get_num_spins(); i++) {
            this->mc_update<total_z_symmetry>(psi, spins, log_psi_real, local_random_state, angle_ptr);
        }
    }

    template<bool total_z_symmetry, typename Psi_t>
    HDINLINE
    void mc_update(const Psi_t& psi, Spins& spins, double& log_psi_real, void* local_random_state, complex_t* angle_ptr) const {
        #ifdef __CUDA_ARCH__

        __shared__ int position;
        __shared__ Spins next_spins;

        if(threadIdx.x == 0) {
            position = curand((curandState_t*)local_random_state) % psi.get_num_spins();
            next_spins = spins.flip(position);
        }
        __syncthreads();

        auto next_angle = psi.flip_spin_of_jth_angle(angle_ptr, threadIdx.x, position, next_spins);

        if(total_z_symmetry) {
            __syncthreads();

            __shared__ int second_position;

            if(threadIdx.x == 0) {
                while(true) {
                    second_position = position = curand((curandState_t*)local_random_state) % psi.get_num_spins();
                    if(next_spins[second_position] == next_spins[position]) {
                        next_spins = next_spins.flip(second_position);
                        break;
                    }
                }
            }
            __syncthreads();
            next_angle = psi.flip_spin_of_jth_angle(&next_angle - threadIdx.x, threadIdx.x, second_position, next_spins);
        }

        __shared__ double next_log_psi_real;
        psi.log_psi_s_real(next_log_psi_real, next_spins, &next_angle - threadIdx.x);

        __shared__ bool spin_flip;

        if(threadIdx.x == 0) {
            const auto ratio = exp(2.0 * (next_log_psi_real - log_psi_real));

            if(ratio > 1.0 || curand_uniform((curandState_t*)local_random_state) <= ratio) {
                spins = next_spins;
                log_psi_real = next_log_psi_real;
                spin_flip = true;
            }
            else {
                spin_flip = false;
            }
        }
        __syncthreads();

        if(spin_flip && threadIdx.x < psi.get_num_angles()) {
            angle_ptr[threadIdx.x] = next_angle;
        }

        #else

        std::uniform_int_distribution<int> random_spin(0, psi.get_num_spins() - 1);
        const auto position = random_spin(*(std::mt19937*)local_random_state);
        Spins next_spins = spins.flip(position);

        complex_t next_angle[Psi_t::get_max_angles()];

        for(auto j = 0u; j < psi.get_num_angles(); j++) {
            next_angle[j] = psi.flip_spin_of_jth_angle(angle_ptr, j, position, next_spins);
        }

        if(total_z_symmetry) {
            int second_position;
            while(true) {
                const auto second_position = random_spin(*(std::mt19937*)local_random_state);

                if(next_spins[second_position] == next_spins[position]) {
                    next_spins = next_spins.flip(second_position);
                    break;
                }
            }
            for(auto j = 0u; j < psi.get_num_angles(); j++) {
                next_angle[j] = psi.flip_spin_of_jth_angle(next_angle, j, second_position, next_spins);
            }
        }

        double next_log_psi_real;
        psi.log_psi_s_real(next_log_psi_real, next_spins, next_angle);
        const auto ratio = exp(2.0 * (next_log_psi_real - log_psi_real));

        std::uniform_real_distribution<double> random_real(0.0, 1.0);
        if(ratio > 1.0 || random_real(*(std::mt19937*)local_random_state) <= ratio) {
            spins = next_spins;
            log_psi_real = next_log_psi_real;
            for(auto j = 0u; j < psi.get_num_angles(); j++) {
                angle_ptr[j] = next_angle[j];
            }
        }

        #endif
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
            const auto blockDim_ = blockDim == -1 ? psi.get_num_angles() : blockDim;

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
