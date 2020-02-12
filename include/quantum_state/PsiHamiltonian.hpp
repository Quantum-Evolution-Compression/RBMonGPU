#pragma once

#include "operator/Operator.hpp"
#include "Array.hpp"
#include "Spins.h"
#include "types.h"
#ifdef __CUDACC__
    #include "utils.kernel"
#endif
#include "cuda_complex.hpp"

#include <vector>
#include <complex>
#include <memory>
#include <cassert>

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pytensor.hpp"

    using namespace std::complex_literals;
#endif // __PYTHONCC__


namespace rbm_on_gpu {

struct PsiHamiltonianAngles {
    PsiHamiltonianAngles() = default;

    template<typename Psi_t>
    HDINLINE void init(const Psi_t& psi, const PsiHamiltonianAngles& other) {
    }

    template<typename Psi_t>
    HDINLINE void init(const Psi_t& psi, const Spins& spins) {
    }

};

namespace kernel {

class PsiHamiltonian {
public:
    unsigned int N;

    static constexpr unsigned int  max_N = MAX_SPINS;

    Operator hamiltonian;

    using Angles = rbm_on_gpu::PsiHamiltonianAngles;

public:

#ifdef __CUDACC__

    HDINLINE
    complex_t log_psi_s(const Spins& spins) const {
        #include "cuda_kernel_defines.h"

        SHARED complex_t energy;
        this->hamiltonian.diagonal_energy(energy, spins);
        return energy;
    }

    HDINLINE
    void log_psi_s(complex_t& result, const Spins& spins, const Angles& angles) const {
        #include "cuda_kernel_defines.h"
        // CAUTION: 'result' has to be a shared variable.
        // j = threadIdx.x

        // result = this->log_psi_s(spins);
        SINGLE {
            result = complex_t(0.0, 0.0);
        }
    }

    HDINLINE
    void log_psi_s_real(double& result, const Spins& spins, const Angles& angles) const {
        // CAUTION: 'result' has to be a shared variable.
        // j = threadIdx.x

        result = this->log_psi_s(spins).real();
    }

    HDINLINE void flip_spin_of_jth_angle(
        const unsigned int j, const unsigned int position, const Spins& new_spins, Angles& angles
    ) const {

    }

    HDINLINE
    complex_t psi_s(const Spins& spins, const Angles& angles) const {
        #include "cuda_kernel_defines.h"

        SHARED complex_t log_psi;
        this->log_psi_s(log_psi, spins, angles);

        return exp(log_psi);
    }

#endif // __CUDACC__

    HDINLINE
    double probability_s(const double log_psi_s_real) const {
        return exp(2.0 * log_psi_s_real);
    }

    HDINLINE
    unsigned int get_num_spins() const {
        return this->N;
    }

    HDINLINE
    unsigned int get_num_angles() const {
        return 0u;
    }

    HDINLINE
    unsigned int get_width() const {
        return this->N;
    }

    PsiHamiltonian get_kernel() const {
        return *this;
    }
};

} // namespace kernel


class PsiHamiltonian : public kernel::PsiHamiltonian {
public:
    bool gpu;

public:
    inline PsiHamiltonian(
        const unsigned int N,
        const Operator& hamiltonian
    ) : gpu(hamiltonian.gpu) {
        this->N = N;
        this->hamiltonian = hamiltonian;
    }

};

} // namespace rbm_on_gpu
