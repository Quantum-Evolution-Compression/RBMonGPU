// #include "QuantumExpression/QuantumExpression.hpp"

// #include "quantum_state/PsiFree.hpp"
// #include "operator/Operator.hpp"


// namespace rbm_on_gpu {

// using namespace std::complex_literals;


// Operator PsiFree::transform_operator(const PauliExpression& expr) const {
//     PauliExpression generator;
//     for(auto i = 0u; i < this->N; i++) {
//         const auto& vec = this->spin_orientations[i];

//         generator += PauliExpression({{i, 0}}, 1i * vec.x);
//         generator += PauliExpression({{i, 1}}, 1i * vec.y);
//         generator += PauliExpression({{i, 2}}, 1i * vec.z);
//     }

//     PauliExpression result_expr(expr);
//     // result_expr.rotate_by(generator);

//     return Operator(result_expr, this->gpu);
// }

// } // namespace rbm_on_gpu
