#pragma once

#include "types.h"
#include <builtin_types.h>
#include <cstdint>
#include <random>
#include <array>

#ifdef __CUDACC__
    #include "curand_kernel.h"
#else
    struct curandState_t;
#endif

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pytensor.hpp"
#endif


namespace rbm_on_gpu {

struct Spins {
    using type = uint64_t;

    #if MAX_SPINS <= 64
        type configuration;
    #elif MAX_SPINS <= 128
        type configuration_first;
        type configuration_second;
    #endif

    Spins() = default;

    #if MAX_SPINS <= 64
    HDINLINE Spins(type configuration, const unsigned int num_spins)
    {
        if(num_spins == 64u) {
            this->configuration = configuration;
        }
        else {
            this->configuration = configuration & (((type)1u << num_spins) - 1u);
        }
    }
    #elif MAX_SPINS <= 128
    // HDINLINE Spins(type configuration_first, type configuration_second)
    //  : configuration_first(configuration_first), configuration_second(configuration_second) {}
    #endif

#ifdef __PYTHONCC__

     decltype(auto) array(const unsigned int num_spins) const {
        xt::pytensor<double, 1u> result(std::array<long int, 1u>({num_spins}));
        for(auto i = 0u; i < num_spins; i++) {
            result[i] = (*this)[i];
        }

        return result;
     }

#endif // __CUDACC__

    HDINLINE static Spins random(void* random_state, const unsigned int num_spins) {
        #ifdef __CUDA_ARCH__
            return Spins(curand(reinterpret_cast<curandState_t*>(random_state)), num_spins);
        #else
            std::uniform_int_distribution<type> random_spin_conf(0, UINT64_MAX);
            return Spins(random_spin_conf(*reinterpret_cast<std::mt19937*>(random_state)), num_spins);
        #endif
    }

    HDINLINE Spins flip(const int position) const {
        return Spins(this->configuration ^ ((type)1 << position), 64u);
    }

    // todo: fix for N = 64
    HDINLINE Spins rotate_left(const unsigned int shift, const unsigned int N) const {
        return Spins(
            (this->configuration << shift) | (this->configuration >> (N - shift)),
            N
        );
    }

    HDINLINE Spins shift_vertical(
        const unsigned int shift, const unsigned int nrows, const unsigned int ncols
    ) const {
        return Spins(
            (this->configuration << (shift * ncols)) | (this->configuration >> ((nrows - shift) * ncols)),
            nrows * ncols
        );
    }

    HDINLINE Spins select_left_columns(const unsigned int select, const unsigned int nrows, const unsigned int ncols) const {
        const auto row = ((1u << select) - 1u) << (ncols - select);
        type mask = 0u;
        for(auto i = 0u; i < nrows; i++) {
            mask |= row << (i * ncols);
        }
        return Spins(this->configuration & mask, nrows * ncols);
    }

    HDINLINE Spins select_right_columns(const unsigned int select, const unsigned int nrows, const unsigned int ncols) const {
        const auto row = (1u << select) - 1u;
        type mask = 0u;
        for(auto i = 0u; i < nrows; i++) {
            mask |= row << (i * ncols);
        }
        return Spins(this->configuration & mask, nrows * ncols);
    }

    HDINLINE Spins shift_horizontal(
        const unsigned int shift, const unsigned int nrows, const unsigned int ncols
    ) const {
        const auto tmp = this->rotate_left(shift, nrows * ncols);
        return Spins(
            (
                tmp.select_left_columns(nrows - shift, nrows, ncols).configuration |
                tmp.select_right_columns(shift, nrows, ncols).shift_vertical(nrows - 1, nrows, ncols).configuration
            ),
            nrows * ncols
        );
    }

    HDINLINE Spins shift_2d(
        const unsigned int shift_i, const unsigned int shift_j,
        const unsigned int nrows, const unsigned int ncols
    ) const {
        return this->shift_vertical(shift_i, nrows, ncols).shift_horizontal(shift_j, nrows, ncols);
    }

    HDINLINE double operator[](const int position) const {
        #if MAX_SPINS <= 64
            return 2.0 * static_cast<double>(
                static_cast<bool>(this->configuration & ((type)1 << position))
            ) - 1.0;
        #elif MAX_SPINS <= 128
            if(position < 64) {
                return 2.0 * static_cast<double>(
                    static_cast<bool>(this->configuration_first & ((type)1 << position))
                ) - 1.0;
            }
            else {
                return 2.0 * static_cast<double>(
                    static_cast<bool>(this->configuration_second & ((type)1 << (position - 64)))
                ) - 1.0;
            }
        #endif
    }

    HDINLINE bool operator==(const Spins& other) const {
        #if MAX_SPINS <= 64
            return this->configuration == other.configuration;
        #elif MAX_SPINS <= 128
            return (this->configuration_first == other.configuration_first) && (this->configuration_second == other.configuration_second);
        #endif
    }

    HDINLINE int total_z(const unsigned int num_spins) const {
        #ifdef __CUDA_ARCH__
            return 2 * __popcll(this->configuration & ((1u << num_spins) - 1)) - num_spins;
        #else
            return 2 * __builtin_popcountll(this->configuration & ((1u << num_spins) - 1)) - num_spins;
        #endif
    }
};

}
