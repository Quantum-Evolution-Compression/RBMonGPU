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
    complex<float> operator()(const Psi_t& psi, const Operator& operator_, const SpinEnsemble& spin_ensemble) const;

    template<typename Psi_t, typename SpinEnsemble>
    vector<complex<float>> operator()(const Psi_t& psi, const vector<Operator>& operator_list_host, const SpinEnsemble& spin_ensemble) const;

    template<typename Psi_t, typename SpinEnsemble>
    pair<float, complex<float>> fluctuation(const Psi_t& psi, const Operator& operator_, const SpinEnsemble& spin_ensemble) const;

    template<typename Psi_t, typename SpinEnsemble>
    complex<float> gradient(complex<float>* result, const Psi_t& psi, const Operator& operator_, const SpinEnsemble& spin_ensemble) const;

    template<typename Psi_t, typename SpinEnsemble>
    void fluctuation_gradient(complex<float>* result, const Psi_t& psi, const Operator& operator_, const SpinEnsemble& spin_ensemble) const;

    template<typename Psi_t, typename SpinEnsemble>
    vector<complex<float>> difference(
        const Psi_t& psi, const Psi_t& psi_prime, const vector<Operator>& operator_list_host, const SpinEnsemble& spin_ensemble
    ) const;

    template<typename Psi_t, typename SpinEnsemble>
    complex<float> __call__(const Psi_t& psi, const Operator& operator_, const SpinEnsemble& spin_ensemble) const {
        return (*this)(psi, operator_, spin_ensemble);
    }

    template<typename Psi_t, typename SpinEnsemble>
    vector<complex<float>> __call__vector(const Psi_t& psi, const vector<Operator>& operator_, const SpinEnsemble& spin_ensemble) const {
        return (*this)(psi, operator_, spin_ensemble);
    }

#ifdef __PYTHONCC__

    template<typename Psi_t, typename SpinEnsemble>
    inline pair<xt::pytensor<complex<float>, 1>, complex<float>> gradient_py(
        const Psi_t& psi, const Operator& operator_, const SpinEnsemble& spin_ensemble
    ) const {
        auto result = xt::pytensor<complex<float>, 1>(
            std::array<long int, 1>({static_cast<long int>(psi.get_num_active_params())})
        );

        const auto expectation_value = this->gradient(result.data(), psi, operator_, spin_ensemble);

        return {result, expectation_value};
    }

    template<typename Psi_t, typename SpinEnsemble>
    inline xt::pytensor<complex<float>, 1> fluctuation_gradient_py(
        const Psi_t& psi, const Operator& operator_, const SpinEnsemble& spin_ensemble
    ) const {
        auto result = xt::pytensor<complex<float>, 1>(
            std::array<long int, 1>({static_cast<long int>(psi.get_num_active_params())})
        );

        this->fluctuation_gradient(result.data(), psi, operator_, spin_ensemble);

        return result;
    }

#endif // __CUDACC__
};

} // namespace rbm_on_gpu
