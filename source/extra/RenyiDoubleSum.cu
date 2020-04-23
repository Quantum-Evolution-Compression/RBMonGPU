#include "extra/RenyiDoubleSum.hpp"
#include "Spins.h"
#include "types.h"
#include "utils.kernel"


namespace rbm_on_gpu {

namespace kernel {

#define TILE_SIZE 256


HDINLINE void renyi_double_sum(double* result, const double* rho_diag, unsigned int N) {
    #include "cuda_kernel_defines.h"

    SHARED double row[TILE_SIZE]; // using a register is not faster
    SHARED double col[TILE_SIZE];

    auto thread_result = 0.0;

    #ifdef __CUDA_ARCH__
    const auto x_offset = blockIdx.x * TILE_SIZE;
    #else
    for(auto x_offset = 0u; x_offset < N; x_offset += TILE_SIZE)
    #endif
    {
        MULTI(i, TILE_SIZE) {
            row[i] = rho_diag[x_offset + i];
        }

        for(auto y_offset = 0u; y_offset < N; y_offset += TILE_SIZE) {
            MULTI(i, TILE_SIZE) {
                col[i] = rho_diag[y_offset + i];
            }
            SYNC;

            MULTI(m, TILE_SIZE) {
                for(auto n = 0u; n < TILE_SIZE; n++) {
                    const auto hamming_distance = bit_count(
                        (x_offset + m) ^ (y_offset + n)
                    );

                    const auto hamming_sign = (hamming_distance & 1u) ? -1.0 : 1.0;
                    const auto hamming_weight = 1.0 / double(1u << hamming_distance);

                    thread_result += hamming_sign * hamming_weight * row[m] * col[n];
                }
            }
            SYNC;
        }
    }

    #ifdef __CUDA_ARCH__

    SHARED double block_result;
    tree_sum(block_result, TILE_SIZE, thread_result);
    SINGLE {
        generic_atomicAdd(result, block_result);
    }

    #else

    *result = thread_result;

    #endif
}


}  // namespace kernel


double renyi_double_sum(const Array<double>& rho_diag) {
    const bool gpu = rho_diag.gpu;
    const auto N = rho_diag.size();

    Array<double> result(1, gpu);
    result.clear();

    auto result_ptr = result.data();
    auto rho_diag_ptr = rho_diag.data();

    if(gpu) {
        cuda_kernel<<<N / TILE_SIZE, TILE_SIZE>>>(
            [=] __device__ () {
                kernel::renyi_double_sum(result_ptr, rho_diag_ptr, N);
            }
        );
    }
    else {
        kernel::renyi_double_sum(result.data(), rho_diag.data(), N);
    }

    result.update_host();
    return result.front();
}


}  // namespace rbm_on_gpu
