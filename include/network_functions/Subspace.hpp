#pragma once

#include "Array.hpp"
#include "types.h"


namespace rbm_on_gpu {

using namespace std;

struct Subspace {

    template<typename Psi_t, typename Psi_t_prime, typename SpinEnsemble>
    complex_t operator()(Array<complex_t>& a_vec, const Psi_t& psi, const Psi_t_prime& psi_prime, SpinEnsemble& spin_ensemble);

    template<typename Psi_t, typename Psi_t_prime, typename SpinEnsemble>
    pair<Array<complex_t>, complex_t> gradient(Array<complex_t>& a_vec, const Psi_t& psi, const Psi_t_prime& psi_prime, SpinEnsemble& spin_ensemble);

};


}  // namespace rbm_on_gpu
