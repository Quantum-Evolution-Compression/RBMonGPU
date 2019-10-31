#pragma once

#include "types.h"
#include <complex>
#include <Array.hpp>

namespace rbm_on_gpu {

using namespace std;


template<typename Psi_t, typename SpinEnsemble>
pair<Array<complex_t>, Array<complex_t>> psi_angles(const Psi_t& psi, const SpinEnsemble& spin_ensemble);

} // namespace rbm_on_gpu
