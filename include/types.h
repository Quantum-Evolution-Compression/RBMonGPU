#pragma once

#include <cuda.h>
#include "cuda_complex.hpp"

#ifdef TIMING
#include <chrono>
#endif

#include <complex>
#include <stdexcept>
#include <string>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <cstring>
#include <array>


namespace rbm_on_gpu {

using namespace std;

using complex_t = cuda_complex::complex<double>;
template<unsigned int dim>
using shape_t = array<long int, dim>;

void setDevice(int device);
void start_profiling();
void stop_profiling();

#ifdef __CUDACC__

#define HDINLINE __host__ __device__ __forceinline__
#define DINLINE __device__ __forceinline__
#define HINLINE __host__ __forceinline__
#define HOST_DEVICE __host__ __device__

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template<typename Function>
__global__ void cuda_kernel(Function function) {
    function();
}

template<typename T>
HDINLINE
void generic_atomicAdd(T* const address, const T& value) {
    #ifdef __CUDA_ARCH__
    atomicAdd(address, value);
    #else
    *address += value;
    #endif
}

#else

#define HDINLINE inline
#define DINLINE inline
#define HINLINE inline
#define HOST_DEVICE

#endif // __CUDACC__

#ifndef MAX_SPINS
#define MAX_SPINS 64
#endif

constexpr auto MAX_HIDDEN_SPINS = 2 * MAX_SPINS;

/**
 * Print a cuda error message including file/line info to stderr
 */
#define PRINT_CUDA_ERROR(msg) \
    std::cerr << "[CUDA] Error: <" << __FILE__ << ">:" << __LINE__ << " " << msg << std::endl

/**
 * Print a cuda error message including file/line info to stderr and raises an exception
 */
#define PRINT_CUDA_ERROR_AND_THROW(cudaError, msg) \
    PRINT_CUDA_ERROR(msg);                         \
    throw std::runtime_error(std::string("[CUDA] Error: ") + std::string(cudaGetErrorString(cudaError)))

#define CUDA_CHECK_KERNEL_CALL {cudaError_t error = cudaGetLastError(); if (error != cudaSuccess) { PRINT_CUDA_ERROR_AND_THROW(error, ""); }}

#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){ PRINT_CUDA_ERROR_AND_THROW(error, ""); }}

#define CUDA_CHECK_MSG(cmd,msg) {cudaError_t error = cmd; if(error!=cudaSuccess){ PRINT_CUDA_ERROR_AND_THROW(error, msg); }}

#define CUDA_CHECK_NO_EXCEP(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){ PRINT_CUDA_ERROR(""); }}

#define CUDA_FREE(x) {if(x != nullptr) {CUDA_CHECK(cudaFree(x)); x = nullptr;}}


#define MALLOC(pointer, size, gpu) {if(gpu) {CUDA_CHECK(cudaMalloc(&pointer, size));} else {pointer = (decltype(pointer)) std::malloc(size);}}
#define MALLOC_MANAGED(pointer, size, gpu) {if(gpu) {CUDA_CHECK(cudaMallocManaged(&pointer, size));} else {pointer = (decltype(pointer)) std::malloc(size);}}
#define FREE(pointer, gpu) { \
    if(pointer != nullptr) { \
        if(gpu) { \
            CUDA_FREE(pointer); \
        } \
        else { \
            std::free(pointer); \
        } \
        pointer = nullptr; \
    } \
}
#define MEMSET(pointer, value, size, gpu) { \
    if(gpu) { \
        CUDA_CHECK(cudaMemset((void*)pointer, value, size)) \
    } else { \
        std::memset((void*)pointer, value, size); \
    } \
}
#define MEMCPY(dest, src, size, dest_on_gpu, src_on_gpu) \
    if(dest_on_gpu) { \
        if(src_on_gpu) { \
            CUDA_CHECK(cudaMemcpy((void*)dest, (void*)src, size, cudaMemcpyDeviceToDevice)) \
        } else { \
            CUDA_CHECK(cudaMemcpy((void*)dest, (void*)src, size, cudaMemcpyHostToDevice)) \
        } \
    } else { \
        if(src_on_gpu) { \
            CUDA_CHECK(cudaMemcpy((void*)dest, (void*)src, size, cudaMemcpyDeviceToHost)) \
        } else { \
            std::memcpy((void*)dest, (void*)src, size); \
        } \
    }
#define MEMCPY_TO_HOST(dest, src, size, gpu) { \
    if(gpu) { \
        CUDA_CHECK(cudaMemcpy((void*)dest, (void*)src, size, cudaMemcpyDeviceToHost)) \
    } else { \
        std::memcpy((void*)dest, (void*)src, size); \
    } \
}

#define ROUND_UP_NEXT_POW2(value) \
        ((value)==1?1:                  \
        ((value)<=2?2:                  \
        ((value)<=4?4:                  \
        ((value)<=8?8:                  \
        ((value)<=16?16:                \
        ((value)<=32?32:                \
        ((value)<=64?64:128             \
        )))))))

/** calculate and set the optimal alignment for data
  *
  * you must align all arrays and structs that are used on the device
  * @param byte size of data in bytes
  */
#define __optimal_align__(byte)                                                \
    alignas(                                                                   \
        ROUND_UP_NEXT_POW2(byte)                                         \
    )

#define CUDA_ALIGN(var,...) __optimal_align__(sizeof(__VA_ARGS__)) __VA_ARGS__ var
#define CUDA_ALIGN8( var, ... ) alignas( 8 ) __VA_ARGS__ var

HDINLINE double sgn(const double& x) {
    #ifdef __CUDA_ARCH__
    return copysign(1.0, x);
    #else
    return x > 0 ? 1 : (x < 0 ? -1 : 0);
    #endif
}

#ifdef TIMING

using namespace std::chrono;
using clock = high_resolution_clock;
std::stringstream* ss;

void log_init() {
    ss = new std::stringstream();
    ss->str("");
}

void log_flush() {
    using namespace std;

    ofstream log_file("/home/burau/timing.txt", ofstream::app);
    log_file << ss->str();
    log_file.flush();
    log_file.close();
    ss->str("");
}

void log_duration(std::string info, std::chrono::duration<double> duration) {
    using namespace std;
    using namespace std::chrono;

    *ss << info << " -> " << duration_cast<microseconds>(duration).count() / 1e3 << " ms" << endl;
    cout << ss->str();
}

#endif

template<typename dtype>
struct std_dtype {
    using type = dtype;
};

template<>
struct std_dtype<complex_t> {
    using type = std::complex<double>;
};


template<typename dtype, typename value_type>
HDINLINE dtype get_real(const value_type& value);

template<>
HDINLINE complex_t get_real<complex_t>(const double& value) {
    return complex_t(value, 0.0);
}

template<>
HDINLINE double get_real<double>(const double& value) {
    return value;
}

template<>
HDINLINE double get_real<double>(const complex_t& value) {
    return value.real();
}

template<typename target_type, typename input_type>
HDINLINE target_type favour_real(const input_type& x);

template<>
HDINLINE complex_t favour_real<complex_t>(const complex_t& x) {
    return x;
}

template<>
HDINLINE double favour_real<double>(const complex_t& x) {
    return x.real();
}

template<>
HDINLINE double favour_real<double>(const double& x) {
    return x;
}


template<unsigned int x>
struct Dim {
    static constexpr unsigned int value = x;
};

} // namespace rbm_on_gpu
