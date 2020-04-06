#pragma once

#include "Spins.h"
#include "Array.hpp"


namespace rbm_on_gpu {

using namespace std;

template<typename Psi_t, typename SpinEnsemble>
Array<complex_t> get_S_matrix(const Psi_t& psi, SpinEnsemble& spin_ensemble);

} // namespace rbm_on_gpu
