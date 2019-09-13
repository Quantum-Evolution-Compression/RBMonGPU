#pragma once

#include "operator/Operator.hpp"
#include "Spins.h"
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
    float*     probability_ratio_avg;
    complex_t*  probability_ratio_O_k_avg;
    float*     next_state_norm_avg;

    complex_t*      local_energies;
    unsigned int    num_local_energies;

    template<bool compute_gradient, typename Psi_t, typename SpinEnsemble>
    void compute_averages(
        const Psi_t& psi, const Psi_t& psi_prime, const Operator& operator_,
        const bool is_unitary, const SpinEnsemble& spin_ensemble
    ) const;
};

} // namespace kernel


class HilbertSpaceDistance : public kernel::HilbertSpaceDistance {
private:

    vector<complex<float>> omega_O_k_avg_host;
    vector<complex<float>> probability_ratio_O_k_avg_host;

public:
    HilbertSpaceDistance(const unsigned int O_k_length, const bool gpu);
    ~HilbertSpaceDistance() noexcept(false);

    void allocate_local_energies(const unsigned int num_local_energies);

    template<typename Psi_t, typename SpinEnsemble>
    float distance(
        const Psi_t& psi, const Psi_t& psi_prime, const Operator& operator_, const bool is_unitary,
        const SpinEnsemble& spin_ensemble
    ) const;

    template<typename Psi_t, typename SpinEnsemble>
    float gradient(
        complex<float>* result, const Psi_t& psi, const Psi_t& psi_prime, const Operator& operator_, const bool is_unitary,
        const SpinEnsemble& spin_ensemble
    );

    template<typename Psi_t, typename SpinEnsemble>
    inline float gradient(
        complex<float>* result, const Psi_t& psi, const Psi_t& psi_prime, const quantum_expression::PauliExpression& expr, const bool is_unitary,
        const SpinEnsemble& spin_ensemble
    ) {
        return this->gradient(
            result, psi, psi_prime, psi.transform_operator(expr), is_unitary, spin_ensemble
        );
    }

#ifdef __PYTHONCC__

    template<typename Psi_t, typename SpinEnsemble>
    pair<xt::pytensor<complex<float>, 1u>, float> gradient_py(
        const Psi_t& psi, const Psi_t& psi_prime, const Operator& operator_, const bool is_unitary,
        const SpinEnsemble& spin_ensemble
    ) {
        xt::pytensor<complex<float>, 1u> grad(std::array<long int, 1u>({(long int)psi_prime.get_num_params()}));

        const float distance = this->gradient(grad.data(), psi, psi_prime, operator_, is_unitary, spin_ensemble);

        return {grad, distance};
    }

#endif // __PYTHONCC__

};

} // namespace rbm_on_gpu
