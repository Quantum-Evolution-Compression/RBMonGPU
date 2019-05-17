#pragma once

#include "operator/Operator.hpp"
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

public:
    ExpectationValue(const bool gpu);
    ~ExpectationValue() noexcept(false);

    template<typename Psi_t, typename SpinEnsemble>
    complex<double> operator()(const Psi_t& psi, const Operator& operator_, const SpinEnsemble& spin_ensemble) const;

    template<typename Psi_t, typename SpinEnsemble>
    vector<complex<double>> operator()(const Psi_t& psi, const vector<Operator>& operator_list_host, const SpinEnsemble& spin_ensemble) const;

    template<typename Psi_t, typename SpinEnsemble>
    pair<double, complex<double>> fluctuation(const Psi_t& psi, const Operator& operator_, const SpinEnsemble& spin_ensemble) const;

    template<typename Psi_t, typename SpinEnsemble>
    void gradient(complex<double>* result, const Psi_t& psi, const Operator& operator_, const SpinEnsemble& spin_ensemble) const;

    template<typename Psi_t, typename SpinEnsemble>
    void fluctuation_gradient(complex<double>* result, const Psi_t& psi, const Operator& operator_, const SpinEnsemble& spin_ensemble) const;

    template<typename Psi_t, typename SpinEnsemble>
    vector<complex<double>> difference(
        const Psi_t& psi, const Psi_t& psi_prime, const vector<Operator>& operator_list_host, const SpinEnsemble& spin_ensemble
    ) const;

    template<typename Psi_t, typename SpinEnsemble>
    complex<double> __call__(const Psi_t& psi, const Operator& operator_, const SpinEnsemble& spin_ensemble) const {
        return (*this)(psi, operator_, spin_ensemble);
    }

    template<typename Psi_t, typename SpinEnsemble>
    vector<complex<double>> __call__vector(const Psi_t& psi, const vector<Operator>& operator_, const SpinEnsemble& spin_ensemble) const {
        return (*this)(psi, operator_, spin_ensemble);
    }

#ifdef __PYTHONCC__

    template<typename Psi_t, typename SpinEnsemble>
    inline xt::pytensor<complex<double>, 1> gradient_py(
        const Psi_t& psi, const Operator& operator_, const SpinEnsemble& spin_ensemble
    ) const {
        auto result = xt::pytensor<complex<double>, 1>(
            std::array<long int, 1>({static_cast<long int>(psi.get_num_active_params())})
        );

        this->gradient(result.raw_data(), psi, operator_, spin_ensemble);

        return result;
    }

    template<typename Psi_t, typename SpinEnsemble>
    inline xt::pytensor<complex<double>, 1> fluctuation_gradient_py(
        const Psi_t& psi, const Operator& operator_, const SpinEnsemble& spin_ensemble
    ) const {
        auto result = xt::pytensor<complex<double>, 1>(
            std::array<long int, 1>({static_cast<long int>(psi.get_num_active_params())})
        );

        this->fluctuation_gradient(result.raw_data(), psi, operator_, spin_ensemble);

        return result;
    }

#endif // __CUDACC__
};

} // namespace rbm_on_gpu
