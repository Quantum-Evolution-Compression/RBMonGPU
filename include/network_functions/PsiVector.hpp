#pragma once

#include <complex>


namespace rbm_on_gpu {

using namespace std;


template<typename Psi_t>
void psi_vector(complex<float>* result, const Psi_t& psi);

} // namespace rbm_on_gpu
