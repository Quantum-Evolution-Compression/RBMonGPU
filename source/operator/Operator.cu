#include "operator/Operator.hpp"
#include <cstring>


namespace rbm_on_gpu {

Operator::Operator(
    const std::vector<std::complex<double>>&        coefficients,
    const std::vector<std::vector<PauliMatrices>>&  pauli_types,
    const std::vector<std::vector<unsigned int>>&   pauli_indices,
    const bool                                      gpu
) : gpu(gpu) {
    using namespace cuda_complex;

    this->num_strings = coefficients.size();
    this->max_string_length = 0u;
    for(const auto& string : pauli_types) {
        if(string.size() > this->max_string_length) {
            this->max_string_length = string.size();
        }
    }
    this->max_string_length++;

    std::vector<PauliMatrices> pauli_types_host(this->max_string_length * this->num_strings);
    std::vector<int> pauli_indices_host(this->max_string_length * this->num_strings);

    for(auto string_index = 0u; string_index < this->num_strings; string_index++) {
        for(auto column = 0u; column < this->max_string_length; column++) {
            const auto table_index = string_index * this->max_string_length + column;
            const auto& pauli_types_string = pauli_types[string_index];
            const auto& pauli_indices_string = pauli_indices[string_index];

            pauli_types_host[table_index] = column < pauli_types_string.size() ? pauli_types_string[column] : PauliMatrices::Identity;
            pauli_indices_host[table_index] = column < pauli_indices_string.size() ? (int)pauli_indices_string[column] : -1;
        }
    }

    this->allocate_memory_and_initialize(
        coefficients.data(), pauli_types_host.data(), pauli_indices_host.data(), false
    );
}


Operator::Operator(const Operator& other) : gpu(other.gpu) {
    this->num_strings = other.num_strings;
    this->max_string_length = other.max_string_length;

    this->allocate_memory_and_initialize(
        reinterpret_cast<complex<double>*>(other.coefficients),
        other.pauli_types,
        other.pauli_indices,
        other.gpu
    );
}

void Operator::allocate_memory_and_initialize(
    const std::complex<double>* coefficients,
    const PauliMatrices*        pauli_types,
    const int*                  pauli_indices,
    const bool                  pointers_on_gpu
) {
    const auto num_table_elements = this->num_strings * this->max_string_length;

    MALLOC(this->coefficients, sizeof(complex_t) * this->num_strings, this->gpu);
    MALLOC(this->pauli_types, sizeof(PauliMatrices) * num_table_elements, this->gpu);
    MALLOC(this->pauli_indices, sizeof(int) * num_table_elements, this->gpu);

    MEMCPY(this->coefficients, coefficients, sizeof(complex_t) * this->num_strings, this->gpu, pointers_on_gpu);
    MEMCPY(this->pauli_types, pauli_types, sizeof(PauliMatrices) * num_table_elements, this->gpu, pointers_on_gpu);
    MEMCPY(this->pauli_indices, pauli_indices, sizeof(int) * num_table_elements, this->gpu, pointers_on_gpu);
}

void Operator::copy_to_host(
    const std::complex<double>* coefficients,
    const PauliMatrices*        pauli_types,
    const int*                  pauli_indices
) const {
    const auto num_table_elements = this->num_strings * this->max_string_length;

    MEMCPY_TO_HOST(coefficients, this->coefficients, sizeof(complex_t) * this->num_strings, this->gpu);
    MEMCPY_TO_HOST(pauli_types, this->pauli_types, sizeof(PauliMatrices) * num_table_elements, this->gpu);
    MEMCPY_TO_HOST(pauli_indices, this->pauli_indices, sizeof(int) * num_table_elements, this->gpu);
}

Operator::~Operator() noexcept(false) {
    FREE(this->coefficients, this->gpu);
    FREE(this->pauli_types, this->gpu);
    FREE(this->pauli_indices, this->gpu);
}

void Operator::get_coefficients(complex<double>* coefficients) const {
    MEMCPY_TO_HOST(coefficients, this->coefficients, sizeof(complex_t) * this->num_strings, this->gpu);
}

void Operator::get_pauli_types(int* pauli_types) const {
    const auto num_table_elements = this->num_strings * this->max_string_length;

    MEMCPY_TO_HOST(pauli_types, this->pauli_types, sizeof(PauliMatrices) * num_table_elements, this->gpu);
}

void Operator::get_pauli_indices(int* pauli_indices) const {
    const auto num_table_elements = this->num_strings * this->max_string_length;

    MEMCPY_TO_HOST(pauli_indices, this->pauli_indices, sizeof(int) * num_table_elements, this->gpu);
}

} // namespace rbm_on_gpu
