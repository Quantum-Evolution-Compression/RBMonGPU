#pragma once

#include "operator/MatrixElement.hpp"
#include "RNGStates.hpp"
#include "random.h"
#include "Array.hpp"
#include "types.h"
#include <vector>

#ifndef __CUDACC__
    #include "QuantumExpression/QuantumExpression.hpp"
#endif


namespace rbm_on_gpu {

namespace kernel {

struct UnitaryChain {
    complex_t*      coefficients;
    int*            pauli_types;
    int*            pauli_indices;

    // table's height
    unsigned int    num_strings;
    // table's width
    unsigned int    max_string_length;

    void* rng_states;
    unsigned int* no_spin_flips;

    HDINLINE
    MatrixElement process_chain(const Spins& spins) const {
        // Important: this is a single-threaded function.

        #include "cuda_kernel_defines.h"
        SHARED MatrixElement result;
        result = {complex_t(1.0, 0.0), spins};

        #ifdef __CUDA_ARCH__
        __shared__ curandState_t rng_state;
        rng_state = reinterpret_cast<curandState_t*>(this->rng_states)[blockIdx.x];
        #else
        mt19937 rng_state = reinterpret_cast<mt19937*>(this->rng_states)[0];
        #endif

        SHARED uint64_t random_bits;
        random_bits = random_uint64(&rng_state);

        SHARED unsigned int n;
        for(n = 0u; n < this->num_strings; n++) {
            SHARED complex_t coefficient;
            coefficient = this->coefficients[n];
            const bool no_spin_flip = this->no_spin_flips[n];
            const bool pq = random_bits & (1u << (n % 64u));

            if(!no_spin_flip) {
                coefficient = pq ? complex_t(0.0, coefficient.imag()) : complex_t(coefficient.real(), 0.0);
            }

            if(pq || no_spin_flip) {
                SHARED unsigned int table_index;
                for(table_index = n * this->max_string_length; true; table_index++) {
                    const auto pauli_index = this->pauli_indices[table_index];
                    if(pauli_index == -1) {
                        break;
                    }

                    if(pauli_index == 1) {
                        result.spins = result.spins.flip(pauli_index);
                    }
                    else if(pauli_index == 2) {
                        coefficient *= complex_t(0.0, -1.0) * result.spins[pauli_index];
                        result.spins = result.spins.flip(pauli_index);
                    }
                    else if(pauli_index == 3) {
                        coefficient *= result.spins[pauli_index];
                    }
                }
            }
            result.coefficient *= coefficient;
        }

        #ifdef __CUDA_ARCH__
        reinterpret_cast<curandState_t*>(this->rng_states)[blockIdx.x] = rng_state;
        #else
        reinterpret_cast<mt19937*>(this->rng_states)[0] = rng_state;
        #endif

        return result;
    }

    template<typename Psi_t>
    HDINLINE
    void local_energy(complex_t& result, const Psi_t& psi, const Spins& spins, const complex_t& log_psi, typename Psi_t::Angles& angles) const {
        #include "cuda_kernel_defines.h"
        // CAUTION: 'result' is only updated by the first thread.

        SHARED MatrixElement matrix_element;

        SINGLE {
            matrix_element = this->process_chain(spins);
            result = matrix_element.coefficient;
        }
        SYNC;
        if(spins != matrix_element.spins) {
            SHARED complex_t log_psi_prime;
            psi.log_psi_s(log_psi_prime, matrix_element.spins, angles);
            SINGLE {
                result *= exp(log_psi_prime - log_psi);
            }
        }
    }

    inline UnitaryChain get_kernel() const {
        return *this;
    }
};

}  // namespace kernel



struct UnitaryChain : public kernel::UnitaryChain {
    Array<complex_t>     coefficients_ar;
    Array<int>           pauli_types_ar;
    Array<int>           pauli_indices_ar;
    Array<unsigned int>  no_spin_flips_ar;
    bool gpu;

#ifndef __CUDACC__
    UnitaryChain(
        const vector<::quantum_expression::PauliExpression>& expr,
        const RNGStates& rng_states,
        const bool gpu
    );
#endif

};

}  // namespace rbm_on_gpu
