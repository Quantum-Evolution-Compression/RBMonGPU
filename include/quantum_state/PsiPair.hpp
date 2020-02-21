#pragma once

#include "quantum_state/PsiDeep.hpp"
#include "quantum_state/psi_functions.hpp"

#include "Array.hpp"
#include "Spins.h"
#include "types.h"
#ifdef __CUDACC__
    #include "utils.kernel"
#endif
#include "cuda_complex.hpp"

#include <vector>
#include <list>
#include <complex>
#include <memory>
#include <cassert>
#include <utility>
#include <algorithm>

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pytensor.hpp"
    #include "xtensor/xcomplex.hpp"

    using namespace std::complex_literals;
#endif // __PYTHONCC__



namespace rbm_on_gpu {

namespace kernel {

struct PsiPair{
    using PsiDeep = PsiDeepT<double>;
    using Angles = PsiDeep::Angles;

    double prefactor;

    PsiDeep psi_real;
    PsiDeep psi_imag;

#ifdef __CUDACC__

    HDINLINE
    void log_psi_s(complex_t& result, const Spins& spins, Angles& cache) const {
        this->psi_real.log_psi_s(result.__re_, spins, cache);
        this->psi_imag.log_psi_s(result.__im_, spins, cache);
    }

    HDINLINE
    void log_psi_s_real(double& result, const Spins& spins, Angles& cache) const {
        this->psi_real.log_psi_s(result, spins, cache);
    }

    HDINLINE void flip_spin_of_jth_angle(
        const unsigned int j, const unsigned int position, const Spins& new_spins, Angles& cache
    ) const {
    }

    HDINLINE
    complex_t psi_s(const Spins& spins, Angles& cache) const {
        #include "cuda_kernel_defines.h"

        SHARED complex_t log_psi;
        this->log_psi_s(log_psi, spins, cache);

        return exp(log(this->prefactor) + log_psi);
    }

    template<typename Function>
    HDINLINE
    void foreach_O_k(const Spins& spins, Angles& cache, Function function) const {
        this->psi_real.foreach_O_k(
            spins,
            cache,
            [&](const unsigned int k, const double& O_k_element) {
                function(k, complex_t(O_k_element, 0.0));
            }
        );
        this->psi_imag.foreach_O_k(
            spins,
            cache,
            [&](const unsigned int k, const double& O_k_element) {
                function(k, complex_t(0.0, O_k_element));
            }
        );
    }

#endif

    HDINLINE
    double probability_s(const double log_psi_s_real) const {
        return exp(2.0 * (log(this->prefactor) + log_psi_s_real));
    }

    HDINLINE
    unsigned int get_num_spins() const {
        return this->psi_real.get_num_spins();
    }

    HDINLINE
    unsigned int get_num_params() const {
        return this->psi_real.get_num_params();
    }

    HDINLINE
    unsigned int get_width() const {
        return this->psi_real.get_width();
    }

    HDINLINE
    unsigned int get_num_angles() const {
        return this->psi_real.get_num_angles();
    }

    HDINLINE
    unsigned int get_num_units() const {
        return this->psi_real.get_num_units();
    }

    HDINLINE
    unsigned int get_O_k_length() const {
        return this->psi_real.get_O_k_length();
    }

    HDINLINE
    PsiPair get_kernel() const {
        return *this;
    }
};

} // namespace kernel


namespace detail {

#ifdef __PYTHONCC__

template<unsigned int dim>
inline vector<xt::pytensor<double, dim>> extract_real(const vector<xt::pytensor<std::complex<double>, dim>>& x_vector) {
    vector<xt::pytensor<double, dim>> result;
    for(const auto& x : x_vector) {
        result.push_back(xt::real(x));
    }
    return result;
}

template<unsigned int dim>
inline vector<xt::pytensor<double, dim>> extract_imag(const vector<xt::pytensor<std::complex<double>, dim>>& x_vector) {
    vector<xt::pytensor<double, dim>> result;
    for(const auto& x : x_vector) {
        result.push_back(xt::imag(x));
    }
    return result;
}

#endif

} // namespace detail


struct PsiPair : public kernel::PsiPair {
    Array<double> alpha_array;
    Array<double> beta_array;
    const bool    free_quantum_axis;

    using PsiDeep = PsiDeepT<double>;
    PsiDeep psi_real;
    PsiDeep psi_imag;

    bool gpu;

    PsiPair(const PsiPair& other);

#ifdef __PYTHONCC__

    inline PsiPair(
        const xt::pytensor<double, 1u>& alpha,
        const xt::pytensor<double, 1u>& beta,
        const vector<xt::pytensor<std::complex<double>, 1u>> biases_list,
        const vector<xt::pytensor<unsigned int, 2u>>& lhs_connections_list,
        const vector<xt::pytensor<std::complex<double>, 2u>>& lhs_weights_list,
        const double prefactor,
        const bool free_quantum_axis,
        const bool gpu
    ) : alpha_array(alpha, false),
        beta_array(beta, false),
        free_quantum_axis(free_quantum_axis),
        psi_real(
            alpha,
            beta,
            detail::extract_real<1u>(biases_list),
            lhs_connections_list,
            detail::extract_real<2u>(lhs_weights_list),
            prefactor,
            free_quantum_axis,
            gpu
        ),
        psi_imag(
            alpha,
            beta,
            detail::extract_imag<1u>(biases_list),
            lhs_connections_list,
            detail::extract_imag<2u>(lhs_weights_list),
            prefactor,
            free_quantum_axis,
            gpu
        ),
        gpu(gpu)
    {
        this->prefactor = prefactor;
        this->update_kernel();
    }

    PsiPair copy() const {
        return *this;
    }

#endif

    Array<complex_t> get_params() const;
    void set_params(const Array<complex_t>& new_params);

    void update_kernel();
};

} // namespace rbm_on_gpu
