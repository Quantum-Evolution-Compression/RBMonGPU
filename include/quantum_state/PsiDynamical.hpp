#pragma once

#include "quantum_state/Psi.hpp"
#include "quantum_state/psi_functions.hpp"
#include "quantum_state/PsiCache.hpp"
#include "spin_ensembles/ExactSummation.hpp"
#include "Spins.h"
#include "types.h"
#include "cuda_complex.hpp"

#include <vector>
#include <complex>
#include <memory>
#include <cassert>

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pytensor.hpp"

    using namespace std::complex_literals;
#endif // __CUDACC__


namespace rbm_on_gpu {

namespace kernel {

class PsiDynamical : public Psi {
public:
    unsigned int* spin_offset_list;
    unsigned int* W_offset_list;
    unsigned int* param_offset_list;
    unsigned int* string_length_list;


#ifdef __CUDACC__
    using Angles = rbm_on_gpu::PsiAngles;
    using Derivatives = rbm_on_gpu::PsiDerivatives;
#endif

public:

#ifdef __CUDACC__

    HDINLINE
    complex_t angle(const unsigned int j, const Spins& spins) const {
        complex_t result = this->b[j];

        const auto W_j = &(this->W[W_offset_list[j]]);
        const auto string_length = this->string_length_list[j];
        const auto spin_offset = this->spin_offset_list[j];
        for(auto i = 0u; i < string_length; i++) {
            result += W_j[i] * spins[(spin_offset + i) % this->N];
        }

        return result;
    }

    HDINLINE void flip_spin_of_jth_angle(
        const unsigned int j, const unsigned int position, const Spins& new_spins, Angles& angles
    ) const {
        // return this->angle(j, new_spins);
        if(j < this->get_num_angles()) {
            const auto relative_position = (position - this->spin_offset_list[j] + this->N) % this->N;
            if(relative_position < this->string_length_list[j]) {
                angles[j] += 2.0 * new_spins[position] * this->W[this->W_offset_list[j] + relative_position];
            }
        }
    }

    template<typename Function>
    HDINLINE
    void foreach_O_k(const Spins& spins, const Angles& angles, Function function) const {
        #include "cuda_kernel_defines.h"

        SHARED Derivatives derivatives;
        derivatives.init(*this, angles);
        SYNC;

        MULTI(j, this->M)
        {
            if(j < this->N) {
                function(j, complex_t(spins[j], 0.0));
            }

            const auto tanh_angle = derivatives.tanh_angles[j];

            function(this->N + j, tanh_angle);

            const auto spin_offset = this->spin_offset_list[j];
            const auto param_offset = this->param_offset_list[j];
            for(auto i = 0u; i < this->string_length_list[j]; i++) {
                function(param_offset + i, tanh_angle * spins[(spin_offset + i) % this->N]);
            }
        }
    }

#endif // __CUDACC__

    inline const PsiDynamical& get_kernel() const {
        return *this;
    }
};

} // namespace kernel


class PsiDynamical : public kernel::PsiDynamical {
public:
    using clist = vector<complex<double>>;

    struct Link {
        unsigned int    first_spin;
        clist           weights;
        complex<double>  hidden_spin_weight;
    };

    clist spin_weights;
    Array<double> alpha_array;
    vector<Link> links;
    vector<pair<int, int>> index_pair_list;
    bool gpu;

public:

#ifdef __PYTHONCC__
    PsiDynamical(const clist& spin_weights, const xt::pytensor<double, 1u>& alpha, const bool gpu=true)
    : spin_weights(spin_weights), alpha_array(alpha, false), gpu(gpu) {
        this->init(spin_weights.data(), spin_weights.size(), false);
    }
#endif

    PsiDynamical(const PsiDynamical& other);
    ~PsiDynamical() noexcept(false);

    void add_hidden_spin(
        const unsigned int first_spin, const clist& link_weights, const complex<double>& hidden_spin_weight
    );
    void update(bool resize);

    inline bool on_gpu() const {
        return this->gpu;
    }

#ifdef __PYTHONCC__
    xt::pytensor<complex<double>, 1> as_vector_py() const {
        auto result = xt::pytensor<complex<double>, 1>(
            std::array<long int, 1>({static_cast<long int>(pow(2, this->N))})
        );
        this->as_vector(result.data());

        return result;
    }

    xt::pytensor<complex<double>, 1> O_k_vector_py(const Spins& spins) const {
        auto result = xt::pytensor<complex<double>, 1>(
            std::array<long int, 1>({static_cast<long int>(this->num_params)})
        );
        this->O_k_vector(result.data(), spins);

        return result;
    }

    xt::pytensor<complex<double>, 1> a_py() const {
        auto result = xt::pytensor<complex<double>, 1>(
            std::array<long int, 1>({static_cast<long int>(this->N)})
        );

        memcpy(result.data(), this->spin_weights.data(), sizeof(complex_t) * this->N);

        return result;
    }

    void set_a_py(const xt::pytensor<complex<double>, 1>& new_a) {
        memcpy(this->spin_weights.data(), new_a.data(), sizeof(complex_t) * this->N);
    }

    xt::pytensor<complex<double>, 1> b_py() const {
        auto result = xt::pytensor<complex<double>, 1>(
            std::array<long int, 1>({static_cast<long int>(this->M)})
        );

        auto j = 0u;
        for(const auto& link : this->links) {
            result.data()[j] = link.hidden_spin_weight;

            j++;
        }

        return result;
    }

    xt::pytensor<complex<double>, 2> dense_W_py() const {
        auto result = xt::pytensor<complex<double>, 2>(
            std::array<long int, 2>(
                {static_cast<long int>(this->N), static_cast<long int>(this->M)}
            )
        );
        this->dense_W(result.data());

        return result;
    }

    xt::pytensor<complex<double>, 1> get_params_py() const {
        auto result = xt::pytensor<complex<double>, 1>(
            std::array<long int, 1>({static_cast<long int>(this->num_params)})
        );
        this->get_params(result.data());

        return result;
    }

    void set_params_py(const xt::pytensor<complex<double>, 1>& new_params) {
        this->set_params(new_params.data());
    }

    PsiDynamical copy() const {
        return *this;
    }

#endif // __PYTHONCC__

    void as_vector(complex<double>* result) const;
    void O_k_vector(complex<double>* result, const Spins& spins) const;
    double norm_function(const ExactSummation& exact_summation) const;
    void dense_W(complex<double>* result) const;

    unsigned int get_num_params_py() const {
        return this->get_num_params();
    }

    void get_params(complex<double>* result) const;
    void set_params(const complex<double>* new_params);

private:

    void init(const complex<double>* a, const unsigned int N, const bool a_on_gpu, const double prefactor=1.0);
    void clear_hidden_spins();
    unsigned int sizeof_W() const;
};

} // namespace rbm_on_gpu
