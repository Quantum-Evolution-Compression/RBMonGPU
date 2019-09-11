#pragma once

#include "spin_ensembles/ExactSummation.hpp"


namespace rbm_on_gpu {

template<typename Psi_t>
float psi_norm(const Psi_t& psi, const ExactSummation& exact_summation);

} // namespace rbm_on_gpu
