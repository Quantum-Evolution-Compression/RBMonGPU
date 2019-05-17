RBMonGPU
========

This library implements the [algorithm of Carleo and Troyer](https://arxiv.org/abs/1606.02318) for respresenting a wavefunction as a restricted Boltzmann machine on the GPU. The code is written in python/C++ and makes heavily use of inline-functions. However it is *not* a header-only library. This means you don't have to compile using the CUDA-compiler for making use of this library within your own code.

Features
--------

 * Monte-Carlo sampling or exact summation of the variational derivatives and any kind of observables on the GPU *or* on the CPU.
 * Same interface for running on GPU and CPU.
 * Update the wavefunction by an arbitrary many-spin-1/2-operator represented as a table of Pauli-matrices.
 * Choose between evaluating the full covariance matrix or by solving the matrix inversion problem iteratively.
 * Various gradient-descent algorithms (momentum, rmpsprop, adamax, ...) for finding the optimal quantum state.
 * Python-interface to all important classes and functions.

Example
-------

```c++
using namespace rbm_on_gpu;

// create 1/2-spin-operator

std::vector<std::complex<double>> coefficients = {1.0+0.5i, 0.9*1.7i, 3.0+0.01i};
std::vector<std::vector<PauliMatrices>> pauli_types = {
    {PauliMatrices::SigmaX, PauliMatrices::SigmaX},
    {PauliMatrices::SigmaZ, PauliMatrices::SigmaX, PauliMatrices::SigmaZ},
    {PauliMatrices::SigmaY}
};
std::vector<std::vector<unsigned int>> pauli_indices = {
    {0, 1},
    {1, 2, 4},
    {3}
};
Operator op(coefficients, pauli_types, pauli_indices);

// create wavefunction

const unsigned int N = 12;
const unsigned int M = 24;
const unsigned int N_params = N + M + N * M;
Psi psi(N, M);

// create monte-carlo algorithm

MonteCarloLoop monte_carlo_loop(N_params, pow(2, 16), 256, M);

/* calculate the derivative of the wavefuntion according to the operator acting on it */

DifferentiatePsi differentiate_psi(monte_carlo_loop, 32);
std::vector<std::complex<double>> dpsi(N_params);

differentiate_psi(psi, op, dpsi);
```

Installation
------------

```
cmake . -DCMAKE_INSTALL_PREFIX=<target-directory>
make install
```

Documentation
-------------

TODO
