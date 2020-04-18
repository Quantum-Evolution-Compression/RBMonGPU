#pragma once

#ifdef ENABLE_MONTE_CARLO
#include "spin_ensembles/MonteCarloLoop.hpp"
#endif // ENABLE_MONTE_CARLO

#ifdef ENABLE_EXACT_SUMMATION
#include "spin_ensembles/ExactSummation.hpp"
#endif // ENABLE_EXACT_SUMMATION

#ifdef ENABLE_SPECIAL_MONTE_CARLO
#include "spin_ensembles/SpecialMonteCarloLoop.hpp"
#endif // ENABLE_SPECIAL_MONTE_CARLO

#ifdef ENABLE_SPECIAL_EXACT_SUMMATION
#include "spin_ensembles/SpecialExactSummation.hpp"
#endif // ENABLE_SPECIAL_EXACT_SUMMATION
