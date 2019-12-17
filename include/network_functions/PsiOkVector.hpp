#pragma once

#include "Spins.h"
#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pytensor.hpp"
#endif // __CUDACC__
#include "Array.hpp"
#include <complex>
#include <utility>


namespace rbm_on_gpu {

using namespace std;


template<typename Psi_t>
void psi_O_k_vector(complex<double>* result, const Psi_t& psi, const Spins& spins);

template<typename Psi_t, typename SpinEnsemble>
void psi_O_k_vector(complex<double>* result, complex<double>* result_std, const Psi_t& psi, const SpinEnsemble& spin_ensemble);


template<typename Psi_t, typename SpinEnsemble>
pair<Array<complex_t>, Array<double>> psi_O_k_vector(const Psi_t& psi, const SpinEnsemble& spin_ensemble);


#ifdef __PYTHONCC__

template<typename Psi_t>
inline xt::pytensor<complex<double>, 1> psi_O_k_vector_py(
    const Psi_t& psi, const Spins& spins
) {
    auto result = xt::pytensor<complex<double>, 1>(
        std::array<long int, 1>({static_cast<long int>(psi.get_O_k_length())})
    );

    psi_O_k_vector(result.data(), psi, spins);

    return result;
}

#endif

} // namespace rbm_on_gpu
