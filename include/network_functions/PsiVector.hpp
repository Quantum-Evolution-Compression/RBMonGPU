#pragma once

#include "types.h"
#include <complex>
#include <Array.hpp>

namespace rbm_on_gpu {

using namespace std;


template<typename Psi_t>
void psi_vector(complex<double>* result, const Psi_t& psi);

template<typename Psi_t>
Array<complex_t> psi_vector(const Psi_t& psi);

} // namespace rbm_on_gpu
