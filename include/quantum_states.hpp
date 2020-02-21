#pragma once

#ifdef ENABLE_PSI
#include "quantum_state/Psi.hpp"
#endif

#ifdef ENABLE_PSI_DEEP
#include "quantum_state/PsiDeep.hpp"
#endif

#ifdef ENABLE_PSI_DEEP_MIN
#include "quantum_state/PsiDeepMin.hpp"
#endif

#ifdef ENABLE_PSI_PAIR
#include "quantum_state/PsiPair.hpp"
#endif

#ifdef ENABLE_PSI_HAMILTONIAN
#include "quantum_state/PsiHamiltonian.hpp"
#endif

#ifdef ENABLE_PSI_CLASSICAL
#include "quantum_state/PsiClassical.hpp"
#endif
