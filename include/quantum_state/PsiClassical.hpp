#pragma once

#include "quantum_state/PsiDeepMin.hpp"
#include "quantum_state/PsiClassicalHelper.hpp"
#include "quantum_state/PsiBase.hpp"


#include "Array.hpp"
#include "Spins.h"
#include "types.h"
#ifdef __CUDACC__
    #include "utils.kernel"
#endif
#include "cuda_complex.hpp"


namespace rbm_on_gpu {

namespace kernel {

struct PsiClassical : public PsiBase {

#ifndef __PYTHONCC__

    HDINLINE
    complex_t log_psi_s(const Spins& spins) const {
         #ifndef __CUDA_ARCH__
         std::vector<int> spins_vector(this->N);
         for(auto i = 0u; i < this->N; i++) {
             spins_vector[i] = spins[i];
         }

         const auto result = Peter::findHeffComplex(spins_vector);
         return complex_t(result.real(), result.imag());

         #else

         return complex_t(0.0, 0.0);

         #endif
    }


    HDINLINE
    void log_psi_s(complex_t& result, const Spins& spins, const Angles& angles) const {
        SINGLE {
            result = this->log_psi_s(spins);
        }
    }

    HDINLINE
    void log_psi_s_real(double& result, const Spins& spins, const Angles& angles) const {
        SINGLE {
            result = this->log_psi_s(spins).real();
        }
    }

    HDINLINE
    complex_t psi_s(const Spins& spins, const Angles& angles) const {
        #include "cuda_kernel_defines.h"

        SHARED complex_t log_psi;
        this->log_psi_s(log_psi, spins, angles);

        return exp(log_psi);
    }

#endif // __PYTHONCC__

    HDINLINE
    double probability_s(const double log_psi_s_real) const {
        return exp(2.0 * log_psi_s_real);
    }

    PsiClassical get_kernel() const {
        return *this;
    }

};

} // namespace kernel


struct PsiClassical : public kernel::PsiClassical, public PsiBase {

    // PsiClassical(const PsiClassical& other);

     inline PsiClassical(
         const string directory,
         const int index,
         const unsigned int N,
         const bool gpu
     ) : rbm_on_gpu::PsiBase(N, false, gpu) {
         this->N = N;

         Peter::load_neural_network(directory, index);
		 Peter::LoadParameters(directory);
         Peter::loadVP(directory, index, "Re");
         Peter::loadVP(directory, index, "Im");
         Peter::Compress_Load(directory, index);

         this->prefactor = 1.0;
    }

};

} // namespace rbm_on_gpu
