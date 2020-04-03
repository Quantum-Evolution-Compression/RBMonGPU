#pragma once

#include "quantum_state/PsiBase.hpp"

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

namespace kernel {

struct PsiExact : public PsiBase {

    complex_t* log_psi_ptr;

#ifndef __PYTHONCC__

    HDINLINE
    complex_t log_psi_s(const Spins& spins) const {
        return this->log_psi_ptr[spins.configuration()];
    }

    HDINLINE
    void log_psi_s(complex_t& result, const Spins& spins, const Angles& angles) const {
        #include "cuda_kernel_defines.h"

        SINGLE {
            result = this->log_psi_s(spins);
        }
        SYNC;
    }

    HDINLINE
    void log_psi_s_real(double& result, const Spins& spins, const Angles& angles) const {
        #include "cuda_kernel_defines.h"

        SINGLE {
            result = this->log_psi_s(spins).real();
        }
        SYNC;
    }

    HDINLINE
    complex_t psi_s(const Spins& spins, const Angles& angles) const {
        #include "cuda_kernel_defines.h"

        SHARED complex_t log_psi;
        this->log_psi_s(log_psi, spins, angles);

        return exp(log(this->prefactor) + log_psi);
    }

#endif // __PYTHONCC__

    HDINLINE
    double probability_s(const double log_psi_s_real) const {
        return exp(2.0 * (log(this->prefactor) + log_psi_s_real));
    }

    HDINLINE
    unsigned int get_num_angles() const {
        return 0u;
    }

    HDINLINE
    unsigned int get_width() const {
        return this->N;
    }

    PsiExact get_kernel() const {
        return *this;
    }

};

} // namespace kernel


class PsiExact : public kernel::PsiExact, public PsiBase {
public:
    Array<complex_t> log_psi_array;


#ifdef __PYTHONCC__

    inline PsiExact(
        const xt::pytensor<std::complex<double>, 1u>& log_psi,
        const unsigned int N,
        const bool gpu
    ) : rbm_on_gpu::PsiBase(N, false, gpu), log_psi_array(log_psi, gpu) {
        this->N = N;
        this->log_psi_array.update_device();
        this->log_psi_ptr = this->log_psi_array.data();

        this->prefactor = 1.0;
    }

    unsigned int get_num_params_py() const {
        return this->get_num_params();
    }

#endif // __PYTHONCC__

};

} // namespace rbm_on_gpu
