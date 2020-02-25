#pragma once

#include "types.h"
#include <complex>
#include <Array.hpp>

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pytensor.hpp"
#endif

namespace rbm_on_gpu {

using namespace std;


template<typename Psi_t>
void psi_vector(complex<double>* result, const Psi_t& psi);

template<typename Psi_t>
Array<complex_t> psi_vector(const Psi_t& psi);


#ifdef __PYTHONCC__

template<typename Psi_t>
inline xt::pytensor<complex<double>, 1u> psi_vector_py(const Psi_t& psi) {
    return psi_vector(psi).to_pytensor_1d();
}

#endif

} // namespace rbm_on_gpu
