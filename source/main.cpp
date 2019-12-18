#define PY_ARRAY_UNIQUE_SYMBOL my_uniqe_array_api_main_cpp

#include "network_functions/ExpectationValue.hpp"
#include "network_functions/HilbertSpaceDistance.hpp"
#include "spin_ensembles/MonteCarloLoop.hpp"
#include "spin_ensembles/ExactSummation.hpp"
#include "quantum_state/Psi.hpp"
#include "cuda_profiler_api.h"

#include <vector>
#include <complex>
#include <iostream>
#include <random>


using namespace std;
using namespace std::complex_literals;
using namespace rbm_on_gpu;


int main(void)
{
    using namespace rbm_on_gpu;

    const auto N = 8u;
    const auto M = N;
    const auto h_max = 1.0;

    std::mt19937 rng(0);
    std::uniform_real_distribution<double> random_real(0.0, 1.0);

    std::vector<std::complex<double>> coefficients;
    std::vector<std::vector<PauliMatrices>> pauli_types;
    std::vector<std::vector<unsigned int>> pauli_indices;

    for(auto i = 0u; i < N; i++) {
        coefficients.push_back(random_real(rng));
        pauli_types.push_back(
            std::vector<PauliMatrices>({
                PauliMatrices::SigmaZ, PauliMatrices::SigmaX, PauliMatrices::SigmaZ
            })
        );
        pauli_indices.push_back(
            std::vector<unsigned int>({
                i, (i + 1) % N
            })
        );

        coefficients.push_back(h_max * random_real(rng));
        pauli_types.push_back(
            std::vector<PauliMatrices>({
                PauliMatrices::SigmaX
            })
        );
        pauli_indices.push_back(
            std::vector<unsigned int>({
                i
            })
        );
    }

    const bool gpu = false;

    Operator op(coefficients, pauli_types, pauli_indices, gpu);
    Psi psi(N, M, 0, 1e-3, false, gpu);

    // DifferentiatePsi differentiate_psi(psi.get_num_params(), gpu);

    // std::vector<std::complex<double>> dpsi(psi.get_num_params());

    ExactSummation exact_summation(N, gpu);
    MonteCarloLoop monte_carlo_loop(
        pow(2, 16), 1, 2, gpu ? 256 : 1, gpu
    );

    // HilbertSpaceDistance hilbert_space_distance(psi.get_num_params(), gpu);

    // cout << result << endl;

    // // // cudaProfilerStart();
    // differentiate_psi(psi, op, monte_carlo_loop, dpsi.data(), 1e-3);

    // const auto values = ExpectationValue(gpu)(psi, vector<Operator>{op, op}, monte_carlo_loop);
    // cout << values[0] << ", " << values[1] << endl;
    // // cudaProfilerStop();

    return 0;
}
