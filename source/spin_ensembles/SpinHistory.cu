#include "spin_ensembles/SpinHistory.hpp"
#include "spin_ensembles/ExactSummation.hpp"
#include "spin_ensembles/MonteCarloLoop.hpp"
#include "quantum_state/Psi.hpp"
#include "quantum_state/PsiDynamical.hpp"
#include "Spins.h"
#include "types.h"


namespace rbm_on_gpu {


SpinHistory::SpinHistory(
    const unsigned int num_steps, const unsigned int num_angles, const bool enable_weights, const bool gpu
) : gpu(gpu) {
    this->num_steps = num_steps;
    this->num_angles = num_angles;

    MALLOC(this->spins, sizeof(Spins) * this->num_steps, this->gpu);
    MALLOC(this->log_psi, sizeof(complex_t) * this->num_steps, this->gpu);
    if(num_angles > 0u) {
        MALLOC(this->angles, sizeof(complex_t) * this->num_steps * num_angles, this->gpu);
    }
    else {
        this->angles = nullptr;
    }

    if(enable_weights) {
        MALLOC(this->weights, sizeof(double) * this->num_steps, this->gpu);
    }
    else {
        this->weights = nullptr;
    }

    this->enable_angles = num_angles > 0u;
}

SpinHistory::~SpinHistory() noexcept(false) {
    FREE(this->spins, this->gpu)
    FREE(this->log_psi, this->gpu)
    FREE(this->angles, this->gpu)
    FREE(this->weights, this->gpu)
}

template<typename Psi_t, typename Generator>
void SpinHistory::fill(const Psi_t& psi, const Generator& generator) const {
    const auto this_kernel = this->get_kernel();

    if(this->gpu) {
        generator.foreach(
            psi,
            [=] __device__ (
                const unsigned int spin_index,
                const Spins spins,
                const complex_t log_psi,
                const complex_t* angle_ptr,
                const double weight
            ) {
                if(this_kernel.enable_angles && threadIdx.x < this_kernel.num_angles) {
                    this_kernel.angles[spin_index * this_kernel.num_angles + threadIdx.x] = angle_ptr[threadIdx.x];
                }

                if(threadIdx.x == 0) {
                    this_kernel.spins[spin_index] = spins;
                    this_kernel.log_psi[spin_index] = log_psi;
                    if(this_kernel.weights != nullptr) {
                        this_kernel.weights[spin_index] = weight;
                    }
                }
            }
        );
    }
    else {
        generator.foreach(
            psi,
            [=] __device__ __host__ (
                const unsigned int spin_index,
                const Spins spins,
                const complex_t log_psi,
                const complex_t* angle_ptr,
                const double weight
            ) {
                if(this_kernel.enable_angles) {
                    for(auto angle_index = 0u; angle_index < this_kernel.num_angles; angle_index++) {
                        this_kernel.angles[spin_index * this_kernel.num_angles + angle_index] = angle_ptr[angle_index];
                    }
                }

                this_kernel.spins[spin_index] = spins;
                this_kernel.log_psi[spin_index] = log_psi;
                if(this_kernel.weights != nullptr) {
                    this_kernel.weights[spin_index] = weight;
                }
            }
        );
    }
}


template void SpinHistory::fill(const Psi&, const ExactSummation&) const;
template void SpinHistory::fill(const Psi&, const MonteCarloLoop&) const;
template void SpinHistory::fill(const PsiDynamical&, const ExactSummation&) const;
template void SpinHistory::fill(const PsiDynamical&, const MonteCarloLoop&) const;

} // namespace rbm_on_gpu
