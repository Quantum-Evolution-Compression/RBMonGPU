#pragma once

#include "MatrixElement.hpp"
#include "Spins.h"
#include "cuda_complex.hpp"
#include "types.h"

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include <xtensor-python/pytensor.hpp>
#endif // __CUDACC__

#ifdef __CUDACC__
    namespace quantum_expression {
        class PauliExpression;
    }
#else
    #include "QuantumExpression/QuantumExpression.hpp"
#endif

#include <vector>
#include <complex>
#include <memory>


namespace rbm_on_gpu {

namespace kernel {

class Operator {
public:
    complex_t*      coefficients;
    PauliMatrices*  pauli_types;
    int*            pauli_indices;

    // table's height
    unsigned int    num_strings;
    // table's width
    unsigned int    max_string_length;

public:

#ifdef __CUDACC__

    template<typename Psi_t>
    HDINLINE
    MatrixElement nth_matrix_element(const Spins& spins, const int n, const Psi_t& psi, typename Psi_t::Angles& angles) const {
        #include "cuda_kernel_defines.h"
        MatrixElement result = {this->coefficients[n], spins};

        for(auto table_index = n * this->max_string_length; true; table_index++) {
            const auto pauli_index = this->pauli_indices[table_index];
            if(pauli_index == -1) {
                break;
            }

            /*
            < s | sigma_x =          < -s |
            < s | sigma_y = -i * s * < -s |
            < s | sigma_z = s *       < s |
            */

            PauliMatrices pauli_type = this->pauli_types[table_index];
            if(pauli_type == PauliMatrices::SigmaX) {
                result.spins = result.spins.flip(pauli_index);

                MULTI(j, psi.get_num_angles())
                {
                    psi.flip_spin_of_jth_angle(j, pauli_index, result.spins, angles);
                }
            }
            else if(pauli_type == PauliMatrices::SigmaY) {
                result.spins = result.spins.flip(pauli_index);

                MULTI(j, psi.get_num_angles())
                {
                    psi.flip_spin_of_jth_angle(j, pauli_index, result.spins, angles);
                }

                result.coefficient *= complex_t(0.0, -1.0) * spins[pauli_index];
            }
            else if(pauli_type == PauliMatrices::SigmaZ) {
                result.coefficient *= spins[pauli_index];
            }
        }

        return result;
    }

    template<typename Psi_t>
    HDINLINE
    void local_energy(complex_t& result, const Psi_t& psi, const Spins& spins, const complex_t& log_psi, const typename Psi_t::Angles& angles) const {
        // CAUTION: 'result' is only updated by the first thread.

        #ifdef __CUDA_ARCH__

        if(threadIdx.x == 0) {
            result = complex_t(0.0, 0.0);
        }

        __shared__ typename Psi_t::Angles angles_prime;

        for(auto n = 0u; n < this->num_strings; n++) {
            angles_prime.init(psi, angles);

            const auto matrix_element = this->nth_matrix_element(
                spins, n, psi, angles_prime
            );

            __shared__ complex_t log_psi_prime;
            psi.log_psi_s(log_psi_prime, matrix_element.spins, angles_prime);
            if(threadIdx.x == 0) {
                result += matrix_element.coefficient * exp(log_psi_prime - log_psi);
            }
        }

        #else

        result = complex_t(0.0, 0.0);
        typename Psi_t::Angles angles_prime;

        for(auto n = 0u; n < this->num_strings; n++) {
            angles_prime.init(psi, angles);

            const auto matrix_element = this->nth_matrix_element(
                spins, n, psi, angles_prime
            );

            complex_t log_psi_prime;
            psi.log_psi_s(log_psi_prime, matrix_element.spins, angles_prime);

            result += matrix_element.coefficient * exp(log_psi_prime - log_psi);
        }

        #endif
    }

    template<typename Psi_t, typename Function>
    HDINLINE
    void foreach_E_k_s_prime(
        const Psi_t& psi, const Spins& spins, const complex_t& log_psi, const typename Psi_t::Angles& angles, Function function
    ) const {
        // E_k = sum_s' E_ss' * psi(s') / psi(s) * O_k(s')
        #include "cuda_kernel_defines.h"

        SHARED typename Psi_t::Angles angles_primes;

        for(auto n = 0u; n < this->num_strings; n++) {
            // #ifdef __CUDA_ARCH__
            // const auto j = threadIdx.x;
            // if(j < psi.get_num_angles())
            // #else
            // for(auto j = 0u; j < psi.get_num_angles(); j++)
            // #endif
            // {
            //     new_angle[j] = angle_ptr[j];
            // }
            angles_primes.init(psi, angles);

            const auto matrix_element = this->nth_matrix_element(
                spins, n, psi, angles_primes
            );

            SHARED complex_t log_psi_prime;
            psi.log_psi_s(log_psi_prime, matrix_element.spins, angles_primes);

            SHARED complex_t E_s_prime;
            SINGLE
            {
                E_s_prime = matrix_element.coefficient * exp(log_psi_prime - log_psi);
            }

            SHARED typename Psi_t::Derivatives derivatives;
            derivatives.init(psi, angles_primes);

            SYNC;

            psi.foreach_O_k(
                matrix_element.spins,
                derivatives,
                [&](const unsigned int k, const complex_t& O_k_element) {
                    function(k, E_s_prime * O_k_element);
                }
            );
        }
    }

#endif // __CUDACC__

    inline Operator get_kernel() const {
        return *this;
    }
};

} // namespace kernel


class Operator : public kernel::Operator {
public:
    bool gpu;

public:

    Operator(
        const std::vector<std::complex<double>>&        coefficients,
        const std::vector<std::vector<PauliMatrices>>&  pauli_types,
        const std::vector<std::vector<unsigned int>>&   pauli_indices,
        const bool                                      gpu=true
    );

    Operator(
        const ::quantum_expression::PauliExpression& expr,
        const bool gpu
    );

    ::quantum_expression::PauliExpression to_expr() const;

#ifdef __PYTHONCC__
    Operator(
        const xt::pytensor<std::complex<double>, 1u>&   coefficients,
        const xt::pytensor<int, 2u>&                    pauli_types,
        const xt::pytensor<int, 2u>&                    pauli_indices,
        const bool                                      gpu
    ) : gpu(gpu) {
        this->num_strings = coefficients.shape()[0];
        this->max_string_length = pauli_types.shape()[1];

        this->allocate_memory_and_initialize(
            coefficients.data(),
            reinterpret_cast<const PauliMatrices*>(pauli_types.data()),
            pauli_indices.data(),
            false
        );
    }

    decltype(auto) get_coefficients_py() const {
        xt::pytensor<complex<double>, 1u> result(array<long int, 1u>({(long int)this->num_strings}));

        this->get_coefficients(result.data());

        return result;
    }
    decltype(auto) get_pauli_types_py() const {
        xt::pytensor<int, 2u> result(array<long int, 2u>({(long int)this->num_strings, (long int)this->max_string_length}));

        this->get_pauli_types(result.data());

        return result;
    }
    decltype(auto) get_pauli_indices_py() const {
        xt::pytensor<int, 2u> result(array<long int, 2u>({(long int)this->num_strings, (long int)this->max_string_length}));

        this->get_pauli_indices(result.data());

        return result;
    }

#endif // __CUDACC__
    Operator(const Operator& other);
    ~Operator() noexcept(false);

private:
    void allocate_memory_and_initialize(
        const std::complex<double>* coefficients,
        const PauliMatrices*        pauli_types,
        const int*                  pauli_indices,
        const bool                  pointers_on_gpu
    );
    void copy_to_host(
        const std::complex<double>* coefficients,
        const PauliMatrices*        pauli_types,
        const int*                  pauli_indices
    ) const;

    void get_coefficients(complex<double>* coefficients) const;
    void get_pauli_types(int* pauli_types) const;
    void get_pauli_indices(int* pauli_indices) const;
};

} // namespace rbm_on_gpu
