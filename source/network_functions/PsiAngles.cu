#include "network_functions/PsiAngles.hpp"
#include "quantum_states.hpp"
#include "spin_ensembles.hpp"
#include "types.h"

namespace rbm_on_gpu {


template<typename Psi_t, typename SpinEnsemble>
pair<Array<complex_t>, Array<complex_t>> psi_angles(const Psi_t& psi, const SpinEnsemble& spin_ensemble) {
    Array<complex_t> result(psi.get_num_units(), psi.gpu);
    Array<complex_t> result_std(psi.get_num_units(), psi.gpu);

    result.clear();
    result_std.clear();

    auto psi_kernel = psi.get_kernel();
    auto result_data = result.data();
    auto result_std_data = result_std.data();

    spin_ensemble.foreach(
        psi,
        [=] __device__ __host__ (
            const unsigned int spin_index,
            const Spins spins,
            const complex_t log_psi,
            typename Psi_t::Angles& angles,
            const double weight
        ) {
            psi_kernel.foreach_angle(spins, angles, [&](const unsigned int j, const complex_t& angle) {
                generic_atomicAdd(&result_data[j], angle);
                generic_atomicAdd(
                    &result_std_data[j],
                    complex_t(
                        angle.real() * angle.real(),
                        angle.imag() * angle.imag()
                    )
                );
            });
        }
    );

    result.update_host();
    result_std.update_host();

    for(auto j = 0u; j < psi.get_num_units(); j++) {
        result[j] /= spin_ensemble.get_num_steps();
        result_std[j] /= spin_ensemble.get_num_steps();

        result_std[j] -= complex_t(
            result[j].real() * result[j].real(),
            result[j].imag() * result[j].imag()
        );
        result_std[j] = complex_t(sqrt(result_std[j].real()), sqrt(result_std[j].imag()));
    }

    return {result, result_std};
}


#ifdef ENABLE_PSI_DEEP

#ifdef ENABLE_EXACT_SUMMATION
template pair<Array<complex_t>, Array<complex_t>> psi_angles(const PsiDeep& psi, const ExactSummation& spin_ensemble);
#endif // ENABLE_EXACT_SUMMATION

#ifdef ENABLE_MONTE_CARLO
template pair<Array<complex_t>, Array<complex_t>> psi_angles(const PsiDeep& psi, const MonteCarloLoop& spin_ensemble);
#endif // ENABLE_MONTE_CARLO

#endif // ENABLE_PSI_DEEP

} // namespace rbm_on_gpu
