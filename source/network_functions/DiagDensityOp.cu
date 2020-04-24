#include "network_functions/DiagDensityOp.hpp"
#include "spin_ensembles.hpp"
#include "quantum_states.hpp"
#include "operator/Operator.hpp"
#include "operator/UnitaryChain.hpp"
#include "cuda_complex.hpp"
#include "types.h"

namespace rbm_on_gpu {


template<typename Psi_t, typename Operator_t, typename SpinEnsemble>
void DiagDensityOp::operator()(const Psi_t& psi, const Operator_t& op, SpinEnsemble& spin_ensemble) {
    this->diag_densities_ar.clear();
    auto diag_densities = this->diag_densities_ar.data();

    const auto psi_kernel = psi.get_kernel();
    const auto op_kernel = op.get_kernel();
    const auto N_A = this->sub_system_size;

    spin_ensemble.foreach(
        psi,
        [=] __device__ __host__ (
            const unsigned int spin_index,
            const Spins& spins,
            const complex_t& log_psi,
            typename Psi_t::Angles& angles,
            const double weight
        ) {
            #include "cuda_kernel_defines.h"

            SHARED complex_t local_energy;

            op_kernel.local_energy(local_energy, psi_kernel, spins, log_psi, angles);

            SINGLE {
                const auto spins_A = spins.extract_first_n(N_A);
                generic_atomicAdd(&diag_densities[spins_A.configuration()], weight * abs2(local_energy));
            }
        }
    );

    this->diag_densities_ar.update_host();
    for(auto& density : this->diag_densities_ar) {
        density /= spin_ensemble.get_num_steps();
    }
}




#ifdef ENABLE_MONTE_CARLO

#ifdef ENABLE_PSI_DEEP

template void DiagDensityOp::operator()(const PsiDeep& psi, const Operator& op, MonteCarloLoop& spin_ensemble);
template void DiagDensityOp::operator()(const PsiDeep& psi, const UnitaryChain& op, MonteCarloLoop& spin_ensemble);

#endif // ENABLE_PSI_DEEP
#ifdef ENABLE_PSI_EXACT

template void DiagDensityOp::operator()(const PsiExact& psi, const Operator& op, MonteCarloLoop& spin_ensemble);
template void DiagDensityOp::operator()(const PsiExact& psi, const UnitaryChain& op, MonteCarloLoop& spin_ensemble);

#endif // ENABLE_PSI_EXACT


#endif // ENABLE_MONTE_CARLO

#ifdef ENABLE_EXACT_SUMMATION

#ifdef ENABLE_PSI_DEEP

template void DiagDensityOp::operator()(const PsiDeep& psi, const Operator& op, ExactSummation& spin_ensemble);
template void DiagDensityOp::operator()(const PsiDeep& psi, const UnitaryChain& op, ExactSummation& spin_ensemble);

#endif // ENABLE_PSI_DEEP
#ifdef ENABLE_PSI_EXACT

template void DiagDensityOp::operator()(const PsiExact& psi, const Operator& op, ExactSummation& spin_ensemble);
template void DiagDensityOp::operator()(const PsiExact& psi, const UnitaryChain& op, ExactSummation& spin_ensemble);

#endif // ENABLE_PSI_EXACT

#endif // ENABLE_EXACT_SUMMATION


}  // namespace rbm_on_gpu
