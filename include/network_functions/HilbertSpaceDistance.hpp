#pragma once

#include "operator/Operator.hpp"
#include "Spins.h"
#include "Array.hpp"
#include "types.h"

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pytensor.hpp"
#endif // __CUDACC__

#include <complex>
// #include <memory>


namespace rbm_on_gpu {

namespace kernel {

class HilbertSpaceDistance {
public:
    bool gpu;

    complex_t*  omega_avg;
    complex_t*  omega_O_k_avg;
    double*     probability_ratio_avg;
    complex_t*  probability_ratio_O_k_avg;
    double*     next_state_norm_avg;

    // free quaxis variables
    double* delta_alpha;
    double* delta_beta;
    double* sin_sum_alpha;
    double* cos_sum_alpha;

    template<bool compute_gradient, bool free_quantum_axis, typename Psi_t, typename SpinEnsemble>
    void compute_averages(
        const Psi_t& psi, const Psi_t& psi_prime, const Operator& operator_,
        const bool is_unitary, const SpinEnsemble& spin_ensemble
    ) const;

    // template<typename Psi_t, typename SpinEnsemble>
    // void overlap(
    //     const Psi_t& psi, const Psi_t& psi_prime, const SpinEnsemble& spin_ensemble
    // ) const;
};

} // namespace kernel


class HilbertSpaceDistance : public kernel::HilbertSpaceDistance {
private:

    vector<complex<double>> omega_O_k_avg_host;
    vector<complex<double>> probability_ratio_O_k_avg_host;

    Array<double> delta_alpha_ar;
    Array<double> delta_beta_ar;
    Array<double> sin_sum_alpha_ar;
    Array<double> cos_sum_alpha_ar;

public:
    HilbertSpaceDistance(const unsigned int N, const unsigned int O_k_length, const bool gpu);
    ~HilbertSpaceDistance() noexcept(false);

    template<typename Psi_t>
    void update_quaxis(const Psi_t& psi, const Psi_t& psi_prime);

    template<typename Psi_t, typename SpinEnsemble>
    double distance(
        const Psi_t& psi, const Psi_t& psi_prime, const Operator& operator_, const bool is_unitary,
        const SpinEnsemble& spin_ensemble
    );

    // template<typename Psi_t, typename SpinEnsemble>
    // double overlap(
    //     const Psi_t& psi, const Psi_t& psi_prime, const SpinEnsemble& spin_ensemble
    // ) const;

    template<typename Psi_t, typename SpinEnsemble>
    double gradient(
        complex<double>* result, const Psi_t& psi, const Psi_t& psi_prime, const Operator& operator_, const bool is_unitary,
        const SpinEnsemble& spin_ensemble
    );

#ifdef __PYTHONCC__

    template<typename Psi_t, typename SpinEnsemble>
    pair<xt::pytensor<complex<double>, 1u>, double> gradient_py(
        const Psi_t& psi, const Psi_t& psi_prime, const Operator& operator_, const bool is_unitary,
        const SpinEnsemble& spin_ensemble
    ) {
        xt::pytensor<complex<double>, 1u> grad(std::array<long int, 1u>({(long int)psi_prime.get_num_params()}));

        const double distance = this->gradient(grad.data(), psi, psi_prime, operator_, is_unitary, spin_ensemble);

        return {grad, distance};
    }

#endif // __PYTHONCC__

};

} // namespace rbm_on_gpu
