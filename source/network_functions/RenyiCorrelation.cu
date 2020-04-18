#include "network_functions/RenyiCorrelation.hpp"
#include "cuda_complex.hpp"
#include "types.h"

namespace rbm_on_gpu {


template<typename Psi_t, typename SpinEnsemble>
double RenyiCorrelation::operator()(const Psi_t& psi, const Operator& U_A, SpinEnsemble& spin_ensemble) {
    this->result_ar.clear();
    auto result = this->result_ar.data();

    const auto psi_kernel = psi.get_kernel();
    const auto U_A_kernel = U_A.get_kernel();

    spin_ensemble.foreach(
        psi,
        [=] __device__ __host__ (
            const unsigned int spin_index,
            const Spins* spins,
            const complex_t* log_psi,
            const typename Psi_t::Angles& angles,
            const double weight
        ) {
            #include "cuda_kernel_defines.h"

            SHARED complex_t local_energy[2];

            U_A_kernel.local_energy(local_energy[0], psi_kernel, spins[0], log_psi[0], angles);
            U_A_kernel.local_energy(local_energy[1], psi_kernel, spins[1], log_psi[1], angles);

            SINGLE {
                generic_atomicAdd(result, weight * abs2(local_energy[0]) * abs2(local_energy[1]));
            }
        }
    );

    this->result_ar.update_host();
    return this->result_ar.front() / spin_ensemble.get_num_steps();
}


}  // namespace rbm_on_gpu
