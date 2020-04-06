#pragma once

#ifdef ENABLE_EXACT_SUMMATION

#include "spin_ensembles/ExactSummation.hpp"


namespace rbm_on_gpu {

template<typename Psi_t>
double psi_norm(const Psi_t& psi, ExactSummation& exact_summation);

} // namespace rbm_on_gpu

#endif // ENABLE_EXACT_SUMMATION
