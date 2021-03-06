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
    unsigned int    num_unitaries;
    // table's width
    unsigned int    max_string_length;

    curandState_t*  rng_states_device;
    mt19937*        rng_states_host;

    unsigned int*   no_spin_flips;
    unsigned int    num_samples;
    bool            fine_sampling;

    HDINLINE
    MatrixElement process_chain(const Spins& spins, void* rng_state) const {
        // Important: this is a single-threaded function.

        #include "cuda_kernel_defines.h"
        SHARED MatrixElement result;
        result = {complex_t(1.0, 0.0), spins};

        SHARED unsigned int n;
        for(n = 0u; n < this->num_unitaries; n++) {
            SHARED uint32_t random_bits;
            if(!this->fine_sampling && n % 32u == 0u) {
                random_bits = random_uint32(rng_state);
            }

            SHARED complex_t coefficient;
            coefficient = this->coefficients[n];
            const bool no_spin_flip = this->no_spin_flips[n];
            const bool pq = (
                this->fine_sampling ?
                // random_real(rng_state) < (coefficient.imag() * coefficient.imag()) :
                random_real(rng_state) < (abs(coefficient.imag()) / (abs(coefficient.real()) + abs(coefficient.imag()))) :
                random_bits & (1u << (n % 32u))
            );

            if(!no_spin_flip) {
                if(this->fine_sampling) {
                    // coefficient = pq ? complex_t(0.0, 1.0 / coefficient.imag()) : complex_t(1.0 / coefficient.real(), 0.0);
                    coefficient = (
                        pq ?
                        complex_t(0.0, coefficient.imag() + sgn(coefficient.imag()) * abs(coefficient.real())) :
                        complex_t(coefficient.real() + sgn(coefficient.real()) * abs(coefficient.imag()), 0.0)
                    );
                }
                else {
                    coefficient = pq ? complex_t(0.0, 2.0 * coefficient.imag()) : complex_t(2.0 * coefficient.real(), 0.0);
                }
            }

            if(no_spin_flip || pq) {
                SHARED unsigned int table_index;
                for(table_index = n * this->max_string_length; true; table_index++) {
                    const auto pauli_index = this->pauli_indices[table_index];
                    if(pauli_index == -1) {
                        break;
                    }
                    const auto pauli_type = this->pauli_types[table_index];

                    if(pauli_type == 1) {
                        result.spins = result.spins.flip(pauli_index);
                    }
                    else if(pauli_type == 2) {
                        coefficient *= complex_t(0.0, -1.0) * result.spins[pauli_index];
                        result.spins = result.spins.flip(pauli_index);
                    }
                    else if(pauli_type == 3) {
                        coefficient.__im_ *= result.spins[pauli_index];
                    }
                }
            }
            result.coefficient *= coefficient;
        }

        return result;
    }

    template<typename Psi_t>
    HDINLINE
    void local_energy(complex_t& result, const Psi_t& psi, const Spins& spins, const complex_t& log_psi, typename Psi_t::Angles& angles) const {
        #include "cuda_kernel_defines.h"
        // CAUTION: 'result' is only updated by the first thread.

        #ifdef __CUDA_ARCH__
        __shared__ curandState_t rng_state;
        SINGLE {
            rng_state = this->rng_states_device[blockIdx.x];
        }
        #else
        mt19937 rng_state = this->rng_states_host[0];
        #endif

        SINGLE {
            result = complex_t(0.0, 0.0);
        }

        SHARED_MEM_LOOP_BEGIN(i, this->num_samples) {

            SHARED complex_t sample;
            SHARED MatrixElement matrix_element;

            SINGLE {
                matrix_element = this->process_chain(spins, &rng_state);
                sample = matrix_element.coefficient;
            }
            SYNC;
            if(spins != matrix_element.spins) {
                SHARED complex_t log_psi_prime;
                psi.log_psi_s(log_psi_prime, matrix_element.spins, angles);
                SINGLE {
                    sample *= exp(log_psi_prime - log_psi);
                }
            }
            SINGLE {
                result += sample;
            }

            SHARED_MEM_LOOP_END(i);
        }
        SINGLE {
            result *= 1.0 / this->num_samples;
            // printf("%f\n", abs2(result));
        }

        #ifdef __CUDA_ARCH__
        SINGLE {
            this->rng_states_device[blockIdx.x] = rng_state;
        }
        #else
        this->rng_states_host[0] = rng_state;
        #endif
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
        const unsigned int num_samples,
        const RNGStates& rng_states,
        const bool gpu
    );
#endif

};

}  // namespace rbm_on_gpu
