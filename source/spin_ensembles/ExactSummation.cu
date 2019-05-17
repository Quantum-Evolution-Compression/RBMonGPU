#include "spin_ensembles/ExactSummation.hpp"
#include "quantum_state/PsiW3.hpp"
#include "quantum_state/Psi.hpp"
#include "types.h"

#include <cassert>


namespace rbm_on_gpu {

ExactSummation::ExactSummation(const unsigned int num_spins)
    : num_spin_configurations(pow(2, num_spins))
    {}

template<typename Psi_t>
ExactSummation::ExactSummation(const Psi_t& psi)
    : num_spin_configurations(pow(2, psi.get_num_spins()))
    {}


template ExactSummation::ExactSummation(const Psi& psi);
template ExactSummation::ExactSummation(const PsiW3& psi);

} // namespace rbm_on_gpu
