#pragma once

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pytensor.hpp"

#endif // __CUDACC__
#include "types.h"
#include <vector>
#include <complex>
#include <iostream>


extern "C" void zminresqlp(
    int* n,
    void (*aprod)(int* n, std::complex<double>* x, std::complex<double>* y),
    std::complex<double>* b,
    double* shift,
    void (*msolve)(int* n, std::complex<double>* x, std::complex<double>* y),
    bool* disable,
    int* nout,
    int* itnlim,
    double* rtol,
    double* maxxnorm,
    double* trancond,
    double* Acondlim,
    std::complex<double>* x,
    int* istop=nullptr,
    int* itn=nullptr,
    double* rnorm=nullptr,
    double* Arnorm=nullptr,
    double* xnorm=nullptr,
    double* Anorm=nullptr,
    double* Acond=nullptr
);


namespace rbm_on_gpu {


inline void solve_Ax_b(
    int n,
    void (*aprod)(int* n, std::complex<double>* x, std::complex<double>* y),
    std::complex<double>* x,
    std::complex<double>* b,
    double rtol
) {
    int nout = 0;

    // std::cout << "hint: " << hint << ", " << (hint == nullptr) << std::endl;

    #ifdef TIMING
    const auto begin = clock::now();
    #endif

    zminresqlp(
        &n,
        aprod,
        b,
        nullptr,
        nullptr,
        nullptr,
        &nout,
        nullptr,
        &rtol,
        nullptr,
        nullptr,
        nullptr,
        x
    );

    #ifdef TIMING
    const auto end = clock::now();

    log_duration("iterative solver", end - begin);
    #endif
}

} // namespace rbm_on_gpu
