#pragma once

#include "quantum_state/Psi.hpp"
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

class HilbertSpaceDistanceV2 {
public:
    bool gpu;

    complex_t* A_avg;
    complex_t* B_avg;
    complex_t* AB_avg;
    double* A2_re_avg;
    double* A2_im_avg;
    double* B2_re_avg;
    double* B2_im_avg;

    complex_t* O_k_avg;
    complex_t* A_O_k_avg;
    double* B_O_k_re_avg;
    double* B_O_k_im_avg;

    template<bool compute_gradient, typename SpinEnsemble>
    void compute_averages(
        const rbm_on_gpu::Psi& psi, const Psi& psi_prime, const Operator& operator_,
        const bool is_unitary, const SpinEnsemble& spin_ensemble
    ) const;
};

} // namespace kernel


class HilbertSpaceDistanceV2 : public kernel::HilbertSpaceDistanceV2 {
private:

public:
    HilbertSpaceDistanceV2(const bool gpu);
    ~HilbertSpaceDistanceV2() noexcept(false);

    template<typename SpinEnsemble>
    double distance(
        const Psi& psi, const Psi& psi_prime, const Operator& operator_, const bool is_unitary,
        const SpinEnsemble& spin_ensemble
    ) const;

};

} // namespace rbm_on_gpu
