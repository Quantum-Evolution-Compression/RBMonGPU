#pragma once

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

struct PsiBaseAngles {
    PsiBaseAngles() = default;

    template<typename Psi_t>
    HDINLINE void init(const Psi_t& psi, const PsiBaseAngles& other) {
    }

    template<typename Psi_t>
    HDINLINE void init(const Psi_t& psi, const Spins& spins) {
    }

};

namespace kernel {

class PsiBase {
public:
    unsigned int N;

    static constexpr unsigned int  max_N = MAX_SPINS;

    // Needs to be initialized by derived class
    unsigned int    num_params;
    unsigned int    O_k_length;

    double          prefactor;

    using Angles = rbm_on_gpu::PsiBaseAngles;


public:

#ifndef __PYTHONCC__

    HDINLINE
    complex_t log_psi_s(const Spins& spins) const {
        return complex_t(0.0, 0.0);
    }



    // HDINLINE
    // complex_t log_psi_s_py(const vector<int>& spins) const {

    //     for(auto n = 0u; n < spins.size(); n++) {

    //     }
    // }

    HDINLINE
    void log_psi_s(complex_t& result, const Spins& spins, const Angles& angles) const {
        #include "cuda_kernel_defines.h"

        SINGLE {
            result = this->log_psi_s(spins);
        }
    }

    HDINLINE
    void log_psi_s_real(double& result, const Spins& spins, const Angles& angles) const {
        #include "cuda_kernel_defines.h"

        SINGLE {
            result = this->log_psi_s(spins).real();
        }
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

#endif // __PYTHONCC__

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

    HDINLINE
    static constexpr unsigned int get_max_spins() {
        return max_N;
    }

    HDINLINE
    static constexpr unsigned int get_max_angles() {
        return max_N;
    }

    HDINLINE
    unsigned int get_num_params() const {
        return this->num_params;
    }

    HDINLINE
    unsigned int get_O_k_length() const {
        return this->O_k_length;
    }

    PsiBase get_kernel() const {
        return *this;
    }

};

} // namespace kernel


struct PsiBase {

    Array<double> alpha_array;
    Array<double> beta_array;

    bool free_quantum_axis;
    bool gpu;

#ifdef __PYTHONCC__

    inline PsiBase(
        const xt::pytensor<double, 1u>& alpha,
        const xt::pytensor<double, 1u>& beta,
        const bool free_quantum_axis,
        const bool gpu
    ) :
        alpha_array(alpha, false),
        beta_array(alpha, false),
        free_quantum_axis(free_quantum_axis),
        gpu(gpu)
    {}

#endif // __PYTHONCC__

    inline PsiBase(
        const Array<double>& alpha,
        const Array<double>& beta,
        const bool free_quantum_axis,
        const bool gpu
    ) :
        alpha_array(alpha),
        beta_array(alpha),
        free_quantum_axis(free_quantum_axis),
        gpu(gpu)
    {}

    inline PsiBase(
        const unsigned int N,
        const bool free_quantum_axis,
        const bool gpu
    ) :
        alpha_array(N, false),
        beta_array(N, false),
        free_quantum_axis(free_quantum_axis),
        gpu(gpu)
    {}

};

} // namespace rbm_on_gpu
