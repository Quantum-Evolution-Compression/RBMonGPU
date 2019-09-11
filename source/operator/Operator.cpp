#define PY_ARRAY_UNIQUE_SYMBOL my_uniqe_array_api_Operator_cpp

#include "operator/Operator.hpp"
#include <algorithm>


namespace rbm_on_gpu {

using namespace quantum_expression;


Operator::Operator(
    const PauliExpression& expr,
    const bool gpu
) : gpu(gpu) {
    this->coefficients = nullptr;
    this->pauli_types = nullptr;
    this->pauli_indices = nullptr;


    this->max_string_length = expr.size() == 0u ? 1 : (
        max_element(
            expr.terms.begin(),
            expr.terms.end(),
            [](const auto& a_term, const auto& b_term) {
                return a_term.first.size() < b_term.first.size();
            }
        )->first.size() + 1
    );

    vector<std::complex<float>> coefficients;
    coefficients.reserve(expr.size());

    vector<PauliMatrices> pauli_types(this->max_string_length * expr.size());
    vector<int> pauli_indices(this->max_string_length * expr.size());

    auto i = 0u;
    for(const auto& term : expr) {
        coefficients.push_back(complex<float>(term.second.real(), term.second.imag()));

        auto j = 0u;
        for(const auto& symbol : term.first.symbols) {
            pauli_types[i * this->max_string_length + j] = PauliMatrices(symbol.op.type);
            pauli_indices[i * this->max_string_length + j] = symbol.index;

            j++;
        }
        for(; j < this->max_string_length; j++) {
            pauli_types[i * this->max_string_length + j] = PauliMatrices::Identity;
            pauli_indices[i * this->max_string_length + j] = -1;
        }

        i++;
    }

    this->num_strings = coefficients.size();

    this->allocate_memory_and_initialize(
        coefficients.data(), pauli_types.data(), pauli_indices.data(), false
    );
}

PauliExpression Operator::to_expr() const {
    vector<std::complex<float>> coefficients(this->num_strings);
    vector<PauliMatrices> pauli_types(this->max_string_length * this->num_strings);
    vector<int> pauli_indices(this->max_string_length * this->num_strings);

    this->copy_to_host(coefficients.data(), pauli_types.data(), pauli_indices.data());

    PauliExpression result;
    for(auto i = 0u; i < this->num_strings; i++) {
        int* pauli_type = reinterpret_cast<int*>(&pauli_types[i * this->max_string_length]);
        int* pauli_index = &pauli_indices[i * this->max_string_length];

        PauliString pauli_string;
        for(;*pauli_index != -1; pauli_type++, pauli_index++) {
            pauli_string.add_symbol({
                *pauli_index, PauliOperator(*pauli_type)
            });
        }
        pauli_string.sort_symbols();

        result += PauliExpression(pauli_string, coefficients[i]);
    }

    return result;
}


} // namespace rbm_on_gpu
