#ifdef ENABLE_SPECIAL_EXACT_SUMMATION

#include "spin_ensembles/SpecialExactSummation.hpp"
#include "quantum_state/Psi.hpp"
#include "types.h"

#include <cassert>
#include <vector>
#include <algorithm>

using namespace std;


namespace rbm_on_gpu {

SpecialExactSummation::SpecialExactSummation(const unsigned int num_spins, const bool gpu)
    :
        gpu(gpu),
        num_spins(num_spins)
    {
        this->num_spin_configurations = pow(2, 2 * num_spins);
    }

} // namespace rbm_on_gpu


#endif // ENABLE_SPECIAL_EXACT_SUMMATION
