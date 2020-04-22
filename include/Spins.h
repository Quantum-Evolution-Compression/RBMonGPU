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

HDINLINE unsigned int bit_count(const uint64_t& x) {
    #ifdef __CUDA_ARCH__
        return __popcll(x);
    #else
        return __builtin_popcountll(x);
    #endif
}

namespace generic {

template<unsigned int num_types>
struct Spins_t {
    using type = uint64_t;
    // constexpr unsigned int num_types_half = num_types / 2u;

    type configurations[num_types];

    Spins_t() = default;

    HDINLINE Spins_t(
        const type configurations[num_types],
        const unsigned int num_spins
    ) {
        #ifdef __CUDA_ARCH__
        #pragma unroll
        #endif
        for(auto i = 0; i < num_types; i++) {
            this->configurations[i] = configurations[i];
        }

        const auto type_idx = num_spins / 64u;
        if(type_idx < num_types) {
            this->configurations[type_idx] &= ((type)1u << (num_spins % 64)) - 1u;
        }
    }

    HDINLINE Spins_t(
        const type configurationsA[num_types / 2u],
        const type configurationsB[num_types / 2u]
    ) {
        #ifdef __CUDA_ARCH__
        #pragma unroll
        #endif
        for(auto i = 0; i < num_types / 2u; i++) {
            this->configurations[i] + configurationsA[i];
            this->configurations[num_types / 2u + i] + configurationsB[i];
        }
    }

    HDINLINE Spins_t(
        const Spins_t<num_types / 2u>& spinsA,
        const Spins_t<num_types / 2u>& spinsB
    ) : Spins_t(spinsA.configurations, spinsB.configurations) {}

    HDINLINE Spins_t& operator=(const Spins_t& other) {
        #ifdef __CUDA_ARCH__
        #pragma unroll
        #endif
        for(auto i = 0; i < num_types; i++) {
            this->configurations[i] = other.configurations[i];
        }

        return *this;
    }

    HDINLINE static Spins_t random(void* random_state, const unsigned int num_spins) {
        // TODO: make it shared?
        Spins_t result;

        #ifdef __CUDA_ARCH__
            #pragma unroll
            for(auto i = 0; i < num_types; i++) {
                result.configurations[i] = curand(reinterpret_cast<curandState_t*>(random_state));
            }
        #else
            for(auto i = 0; i < num_types; i++) {
                std::uniform_int_distribution<type> random_spin_conf(0, UINT64_MAX);
                result.configurations[i] = random_spin_conf(*reinterpret_cast<std::mt19937*>(random_state));
            }
        #endif

        const auto type_idx = num_spins / 64u;
        if(type_idx < num_types) {
            result.configurations[type_idx] &= ((type)1u << (num_spins % 64)) - 1u;
        }

        return result;
    }

    HDINLINE static Spins_t all_up(const unsigned int num_spins) {
        Spins_t result;

        #ifdef __CUDA_ARCH__
        #pragma unroll
        #endif
        for(auto i = 0; i < num_types; i++) {
            result.configurations[i] = ~type(0);
        }

        const auto type_idx = num_spins / 64u;
        if(type_idx < num_types) {
            result.configurations[type_idx] &= ((type)1u << (num_spins % 64)) - 1u;
        }

        return result;
    }


    HDINLINE double operator[](const unsigned int position) const {
        return 2.0 * static_cast<double>(
            static_cast<bool>(this->configurations[position / 64u] & ((type)1 << (position % 64u)))
        ) - 1.0;
    }

    #ifdef __PYTHONCC__

    decltype(auto) array(const unsigned int num_spins) const {
        xt::pytensor<double, 1u> result(std::array<long int, 1u>({num_spins}));
        for(auto i = 0u; i < num_spins; i++) {
            result[i] = (*this)[i];
        }

        return result;
    }

    #endif // __CUDACC__

    HDINLINE Spins_t flip(const unsigned int position) const {
        Spins_t result = *this;
        result.configurations[position / 64u] ^= ((type)1 << (position % 64u));
        return result;
    }

    HDINLINE bool operator==(const Spins_t& other) const {
        #ifdef __CUDA_ARCH__
        #pragma unroll
        #endif
        for(auto i = 0; i < num_types; i++) {
            if(this->configurations[i] != other.configurations[i]) {
                return false;
            }
        }
        return true;
    }

    HDINLINE Spins_t& operator^=(const Spins_t& other) {
        #ifdef __CUDA_ARCH__
        #pragma unroll
        #endif
        for(auto i = 0; i < num_types; i++) {
            this->configurations[i] ^= other.configurations[i];
        }

        return *this;
    }

    HDINLINE Spins_t& operator&=(const Spins_t& other) {
        #ifdef __CUDA_ARCH__
        #pragma unroll
        #endif
        for(auto i = 0; i < num_types; i++) {
            this->configurations[i] &= other.configurations[i];
        }

        return *this;
    }

    HDINLINE Spins_t& operator|=(const Spins_t& other) {
        #ifdef __CUDA_ARCH__
        #pragma unroll
        #endif
        for(auto i = 0; i < num_types; i++) {
            this->configurations[i] |= other.configurations[i];
        }

        return *this;
    }


    HDINLINE int total_z(const unsigned int num_spins) const {
        auto result = 0;

        for(auto i = 0; i < num_types - 1u; i++) {
            #ifdef __CUDA_ARCH__
                result += 2 * bit_count(this->configurations[i]) - 64;
            #else
                result += 2 * bit_count(this->configurations[i]) - 64;
            #endif
        }
        const auto type_idx = num_spins / 64u;
        if(type_idx < num_types) {
            #ifdef __CUDA_ARCH__
                result += 2 * bit_count(this->configurations[num_types - 1u] & ((1u << (num_spins % 64u)) - 1)) - (num_spins % 64u);
            #else
                result += 2 * bit_count(this->configurations[num_types - 1u] & ((1u << (num_spins % 64u)) - 1)) - (num_spins % 64u);
            #endif
        }

        return result;
    }
};

}  // namespace generic


template<unsigned int num_types>
struct Spins_t : public generic::Spins_t<num_types> {};

template<>
struct Spins_t<1u> : public generic::Spins_t<1u> {

    Spins_t<1u>() = default;

    HDINLINE Spins_t<1u>(type configuration, const unsigned int num_spins) {
        if(num_spins == 64u) {
            this->configuration() = configuration;
        }
        else {
            this->configuration() = configuration & (((type)1u << num_spins) - 1u);
        }
    }

    HDINLINE type& configuration() {
        return this->configurations[0];
    }

    HDINLINE const type& configuration() const {
        return this->configurations[0];
    }

    HDINLINE unsigned int hamming_distance(const Spins_t<1u>& other) const {
        return bit_count(this->configuration() ^ other.configuration());
    }

    HDINLINE uint64_t bit_at(const unsigned int i) const {
        return this->configuration() & ((type)1u << i);
    }

    HDINLINE Spins_t<1u> extract_first_n(const unsigned int n) const {
        return Spins_t<1u>(this->configuration(), n);
    }

    HDINLINE Spins_t<1u>& operator=(const Spins_t<1u>& other) {
        this->configuration() = other.configuration();

        return *this;
    }

    HDINLINE Spins_t<1u>& operator=(const generic::Spins_t<1u>& other) {
        this->configuration() = other.configurations[0];

        return *this;
    }

    // todo: fix for N = 64
    HDINLINE Spins_t<1u> rotate_left(const unsigned int shift, const unsigned int N) const {
        return Spins_t<1u>(
            (this->configuration() << shift) | (this->configuration() >> (N - shift)),
            N
        );
    }

    HDINLINE Spins_t<1u> shift_vertical(
        const unsigned int shift, const unsigned int nrows, const unsigned int ncols
    ) const {
        return Spins_t<1u>(
            (this->configuration() << (shift * ncols)) | (this->configuration() >> ((nrows - shift) * ncols)),
            nrows * ncols
        );
    }

    HDINLINE Spins_t<1u> select_left_columns(const unsigned int select, const unsigned int nrows, const unsigned int ncols) const {
        const auto row = ((1u << select) - 1u) << (ncols - select);
        type mask = 0u;
        for(auto i = 0u; i < nrows; i++) {
            mask |= row << (i * ncols);
        }
        return Spins_t<1u>(this->configuration() & mask, nrows * ncols);
    }

    HDINLINE Spins_t<1u> select_right_columns(const unsigned int select, const unsigned int nrows, const unsigned int ncols) const {
        const auto row = (1u << select) - 1u;
        type mask = 0u;
        for(auto i = 0u; i < nrows; i++) {
            mask |= row << (i * ncols);
        }
        return Spins_t<1u>(this->configuration() & mask, nrows * ncols);
    }

    HDINLINE Spins_t<1u> shift_horizontal(
        const unsigned int shift, const unsigned int nrows, const unsigned int ncols
    ) const {
        const auto tmp = this->rotate_left(shift, nrows * ncols);
        return Spins_t<1u>(
            (
                tmp.select_left_columns(nrows - shift, nrows, ncols).configuration() |
                tmp.select_right_columns(shift, nrows, ncols).shift_vertical(nrows - 1, nrows, ncols).configuration()
            ),
            nrows * ncols
        );
    }

    HDINLINE Spins_t<1u> shift_2d(
        const unsigned int shift_i, const unsigned int shift_j,
        const unsigned int nrows, const unsigned int ncols
    ) const {
        return this->shift_vertical(shift_i, nrows, ncols).shift_horizontal(shift_j, nrows, ncols);
    }
};


#if MAX_SPINS <= 64
using Spins = Spins_t<1u>;
#elif MAX_SPINS <= 128
using Spins = Spins_t<2u>;
#endif


}  // namespace rbm_on_gpu
