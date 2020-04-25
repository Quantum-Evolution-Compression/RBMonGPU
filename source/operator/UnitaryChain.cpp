#define PY_ARRAY_UNIQUE_SYMBOL my_uniqe_array_api_UnitaryChain_cpp

#include "operator/UnitaryChain.hpp"
#include <algorithm>
#include <math.h>
#include <iostream>


namespace rbm_on_gpu {

using namespace quantum_expression;
using namespace std;


UnitaryChain::UnitaryChain(
    const vector<::quantum_expression::PauliExpression>& expr_chain,
    const unsigned int num_samples,
    const RNGStates& rng_states,
    const bool gpu
) : coefficients_ar(gpu),
    pauli_types_ar(gpu),
    pauli_indices_ar(gpu),
    no_spin_flips_ar(gpu),
    gpu(gpu) {

    this->max_string_length = 1u;
    for(const auto& expr : expr_chain) {
        for(const auto& terms : expr) {
            if(terms.first.size() + 1u > this->max_string_length) {
                this->max_string_length = terms.first.size() + 1u;
            }
        }
    }

    this->coefficients_ar.resize(expr_chain.size());
    this->pauli_types_ar.resize(this->max_string_length * expr_chain.size());
    this->pauli_indices_ar.resize(this->max_string_length * expr_chain.size());
    this->no_spin_flips_ar.resize(expr_chain.size());

    auto i = 0u;
    for(const auto& expr : expr_chain) {

        bool no_spin_flip_here = true;
        auto j = 0u;
        for(const auto& symbol : expr.begin()->first.symbols) {
            this->pauli_types_ar[i * this->max_string_length + j] = symbol.op.type;
            this->pauli_indices_ar[i * this->max_string_length + j] = symbol.index;

            if(symbol.op.type == 1 || symbol.op.type == 2) {
                no_spin_flip_here = false;
            }

            j++;
        }
        for(; j < this->max_string_length; j++) {
            this->pauli_types_ar[i * this->max_string_length + j] = 1;
            this->pauli_indices_ar[i * this->max_string_length + j] = -1;
        }

        this->no_spin_flips_ar[i] = no_spin_flip_here;
        this->coefficients_ar[i] = complex<double>(
            cos(expr.begin()->second.imag()),
            sin(expr.begin()->second.imag())
        );

        // this compensates for the stochastic chance of 1/2 to choose either the Pauli-string or the scalar value.
        if(!no_spin_flip_here) {
            this->coefficients_ar[i] *= 2.0;
        }

        i++;
    }

    this->num_unitaries = this->coefficients_ar.size();

    this->coefficients_ar.update_device();
    this->pauli_types_ar.update_device();
    this->pauli_indices_ar.update_device();
    this->no_spin_flips_ar.update_device();

    this->coefficients = this->coefficients_ar.data();
    this->pauli_types = this->pauli_types_ar.data();
    this->pauli_indices = this->pauli_indices_ar.data();
    this->no_spin_flips = this->no_spin_flips_ar.data();

    this->num_samples = num_samples;
    this->rng_states_device = rng_states.rng_states_device;
    this->rng_states_host = rng_states.rng_states_host;
}


} // namespace rbm_on_gpu
