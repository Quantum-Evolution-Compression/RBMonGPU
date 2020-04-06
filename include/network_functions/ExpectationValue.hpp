#pragma once

#include "operator/Operator.hpp"
#include "Array.hpp"
#include "types.h"

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pytensor.hpp"
#endif // __CUDACC__

#include <vector>
#include <complex>

namespace rbm_on_gpu {


class ExpectationValue {
private:
    bool        gpu;
    complex_t*  result;

    Array<complex_t> A_local;
    Array<double> A_local_abs2;

public:
    ExpectationValue(const bool gpu);
    ~ExpectationValue() noexcept(false);

    template<typename Psi_t, typename SpinEnsemble>
    complex<double> operator()(const Psi_t& psi, const Operator& operator_, SpinEnsemble& spin_ensemble) const;

    template<typename Psi_t, typename SpinEnsemble>
    pair<complex<double>, double> corrected(const Psi_t& psi, const Operator& operator_, const Operator& operator2, SpinEnsemble& spin_ensemble);

    template<typename Psi_t, typename SpinEnsemble>
    vector<complex<double>> operator()(const Psi_t& psi, const vector<Operator>& operator_list_host, SpinEnsemble& spin_ensemble) const;

    template<typename Psi_t, typename SpinEnsemble>
    pair<double, complex<double>> fluctuation(const Psi_t& psi, const Operator& operator_, SpinEnsemble& spin_ensemble) const;

    template<typename Psi_t, typename SpinEnsemble>
    complex<double> gradient(complex<double>* result, const Psi_t& psi, const Operator& operator_, SpinEnsemble& spin_ensemble) const;

    template<typename Psi_t, typename SpinEnsemble>
    void fluctuation_gradient(complex<double>* result, const Psi_t& psi, const Operator& operator_, SpinEnsemble& spin_ensemble) const;

    template<typename Psi_t, typename SpinEnsemble>
    vector<complex<double>> difference(
        const Psi_t& psi, const Psi_t& psi_prime, const vector<Operator>& operator_list_host, SpinEnsemble& spin_ensemble
    ) const;

    template<typename Psi_t, typename SpinEnsemble>
    complex<double> __call__(const Psi_t& psi, const Operator& operator_, SpinEnsemble& spin_ensemble) const {
        return (*this)(psi, operator_, spin_ensemble);
    }

    template<typename Psi_t, typename SpinEnsemble>
    vector<complex<double>> __call__vector(const Psi_t& psi, const vector<Operator>& operator_, SpinEnsemble& spin_ensemble) const {
        return (*this)(psi, operator_, spin_ensemble);
    }

#ifdef __PYTHONCC__

    template<typename Psi_t, typename SpinEnsemble>
    inline pair<xt::pytensor<complex<double>, 1>, complex<double>> gradient_py(
        const Psi_t& psi, const Operator& operator_, SpinEnsemble& spin_ensemble
    ) const {
        auto result = xt::pytensor<complex<double>, 1>(
            std::array<long int, 1>({static_cast<long int>(psi.get_num_params())})
        );

        const auto expectation_value = this->gradient(result.data(), psi, operator_, spin_ensemble);

        return {result, expectation_value};
    }

    template<typename Psi_t, typename SpinEnsemble>
    inline xt::pytensor<complex<double>, 1> fluctuation_gradient_py(
        const Psi_t& psi, const Operator& operator_, SpinEnsemble& spin_ensemble
    ) const {
        auto result = xt::pytensor<complex<double>, 1>(
            std::array<long int, 1>({static_cast<long int>(psi.get_num_params())})
        );

        this->fluctuation_gradient(result.data(), psi, operator_, spin_ensemble);

        return result;
    }

#endif // __CUDACC__
};

} // namespace rbm_on_gpu
