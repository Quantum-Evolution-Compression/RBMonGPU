#pragma once

#ifdef ENABLE_PSI
#include "quantum_state/Psi.hpp"
#endif // ENABLE_PSI

#ifdef ENABLE_PSI_DEEP
#include "quantum_state/PsiDeep.hpp"
#endif // ENABLE_PSI_DEEP

#ifdef ENABLE_PSI_DEEP_MIN
#include "quantum_state/PsiDeepMin.hpp"
#endif // ENABLE_PSI_DEEP_MIN

#ifdef ENABLE_PSI_PAIR
#include "quantum_state/PsiPair.hpp"
#endif // ENABLE_PSI_PAIR

#ifdef PSI_HAMILTONIAN
#include "quantum_state/PsiHamiltonian.hpp"
#endif // PSI_HAMILTONIAN

#ifdef ENABLE_PSI_CLASSICAL
#include "quantum_state/PsiClassical.hpp"
#endif // ENABLE_PSI_CLASSICAL

#ifdef ENABLE_PSI_EXACT
#include "quantum_state/PsiExact.hpp"
#endif // ENABLE_PSI_EXACT
