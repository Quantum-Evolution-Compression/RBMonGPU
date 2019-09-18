#pragma once

#include "quantum_state/psi_functions.hpp"
#include "quantum_state/PsiDeepDeepCache.hpp"
#include "Array.hpp"
#include "Spins.h"
#include "types.h"
#ifdef __CUDACC__
    #include "utils.kernel"
#endif
#include "cuda_complex.hpp"

#include <vector>
#include <complex>
#include <memory>
#include <cassert>
#include <utility>

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pytensor.hpp"

    using namespace std::complex_literals;
#endif // __PYTHONCC__


namespace rbm_on_gpu {

namespace kernel {

class PsiDeep {
public:
    using Angles = rbm_on_gpu::PsiDeepAngles;

    constexpr unsigned int max_layers = 5u;
    constexpr unsigned int max_deep_angles = max_layers * MAX_SPINS;

    struct Layer {
        unsigned int  size;                 // number of units
        unsigned int  lhs_connectivity;     // number of connections to the lhs per unit
        unsigned int  rhs_connectivity;     // number of connections to the rhs per unit
        unsigned int  delta;                // step-width of connections to the lhs
        unsigned int  begin_angles;         // index of angle corresponding to the first unit of this layer in a global list of angles
        unsigned int  begin_params;         // index of param corresponding to the first unit of this layer in a global list of params
        complex_t*    lhs_weights;          // weight matrix to the lhs: lhs-connectivity x size
        complex_t*    rhs_weights;          // weight matrix to the rhs: size x rhs-connectivity
        complex_t*    bases;                // basis factors
        unsigned int* rhs_connections;      // connectivity matrix to the rhs: size x rhs-connectivity

        HDINLINE unsigned int lhs_connection(const unsigned int i) const {
            return this->delta * i;
        }
    };

    unsigned int   N;
    complex_t*     a;
    Layer          layers[max_layers];
    unsigned int   num_layers;

    unsigned int   num_params;
    float          prefactor;

public:

#ifdef __CUDACC__

    HDINLINE
    void forward_pass(const Spins& spins, complex_t* activations_out, complex_t* deep_angles) {
        #include "cuda_kernel_defines.h"

        SHARED complex_t activations_in[Angles::max_width];
        MULTI(i, this->get_num_spins()) {
            if(this->num_layers % 2u == 1u) {
                activations_in[i] = spins[i];
            } else {
                activations_out[i] = spins[i];
            }
        }
        SYNC;

        for(auto layer_idx = 0u; layer_idx < this->num_layers; layer_idx++) {
            if((layer_idx > 0u) || (this->num_layers % 2u == 0u)) {
                SYNC;
                SINGLE {
                    complex_t* tmp = activations_out;
                    activations_out = activations_in;
                    activations_in = tmp;
                }
                SYNC;
            }
            const Layer& layer = this->layers[layer_idx];
            MULTI(j, layer.size) {
                activations_out[j] = complex_t(0.0f, 0.0f);
                for(auto i = 0u; i < layer.lhs_connectivity; i++) {
                    activations_out[j] += (
                        layer.lhs_weights[i * layer.size + j] *
                        activations_in[
                            (layer.lhs_connection(j) + i) % (
                                layer_idx == 0u ?
                                this->N :
                                this->layers[layer_idx - 1u].size
                            )
                        ]
                    );
                }
                if(deep_angles != nullptr) {
                    deep_angles[layer.begin_angles + j] = activations_out[j];
                }
                activations_out[j] = my_logcosh(activations_out[j]);
            }
        }
    }

    HDINLINE
    void log_psi_s(complex_t& result, const Spins& spins, const Angles& cache) const {
        // CAUTION: 'result' has to be a shared variable.

        this->forward_pass(spins, cache.activations, nullptr);
        const auto final_layer_size = this->layers[this->num_layers - 1u].size;
        MULTI(j, max(this->N, final_layer_size)) {
            auto summand = (
                (j < this->N ? this->a[j] * spins[j] : complex_t(0.0f, 0.0f)) +
                (j < final_layer_size ? cache.activations[j] : complex_t(0.0f, 0.0f))
            );

            tree_sum(result, max(this->N, final_layer_size), summand);
        }
    }

    HDINLINE
    void log_psi_s_real(float& result, const Spins& spins, const Angles& cache) const {
        // CAUTION: 'result' has to be a shared variable.

        this->forward_pass(spins, cache.activations, nullptr);
        const auto final_layer_size = this->layers[this->num_layers - 1u].size;
        MULTI(j, max(this->N, final_layer_size)) {
            auto summand = float(
                (j < this->N ? this->a[j] * spins[j] : float(0.0f)) +
                (j < final_layer_size ? cache.activations[j] : float(0.0f))
            );

            tree_sum(result, max(this->N, final_layer_size), summand);
        }
    }

    HDINLINE void flip_spin_of_jth_angle(
        const unsigned int j, const unsigned int position, const Spins& new_spins, Angles& cache
    ) const {
    }

    HDINLINE
    complex_t psi_s(const Spins& spins, const Angles& cache) const {
        #include "cuda_kernel_defines.h"

        SHARED complex_t log_psi;
        this->log_psi_s(log_psi, spins, cache);

        return exp(log(this->prefactor) + log_psi);
    }

    template<typename Function>
    HDINLINE
    void foreach_O_k(const Spins& spins, const Angles& cache, Function function) const {
        #include "cuda_kernel_defines.h"

        MULTI(i, this->N) {
            function(i, complex_t(spins[i], 0.0f));
        }

        SHARED complex_t deep_angles[this->max_deep_angles];
        this->forward_pass(spins, cache.activations, deep_angles);

        for(int layer_idx = this->num_layers - 1; layer_idx >= 0; layer_idx--) {
            const Layer& layer = this->layers[layer_idx];

            // calculate the unit-activations of the layer.
            // here, these are the back-propagated derivatives.
            if(layer_idx == this->num_layers - 1) {
                MULTI(j, layer.size) {
                    cache.activations[j] = my_tanh(deep_angles[
                        layer.begin_angles + j
                    ]);
                }
            } else {
                // TODO: check if shared memory solution is faster
                #ifdef __CUDA_ARCH__
                complex_t unit_activation(0.0f, 0.0f);
                #else
                complex_t unit_activation[max_width];
                #endif

                SYNC;
                MULTI(i, layer.size) {
                    #ifndef __CUDA_ARCH__
                    unit_activation[i] = complex_t(0.0f, 0.0f);
                    #endif

                    for(auto j = 0u; j < layer.rhs_connectivity; j++) {
                        #ifdef __CUDA_ARCH__
                        unit_activation +=
                        #else
                        unit_activation[i] +=
                        #endif
                        (
                            layer.rhs_weights[i * layer.rhs_connectivity + j] * cache.activations[
                                layer.rhs_connections[i * layer.rhs_connectivity + j]
                            ]
                        );
                    }
                    #ifdef __CUDA_ARCH__
                    unit_activation *=
                    #else
                    unit_activation[i] *=
                    #endif
                    (
                        my_tanh(deep_angles[layer.begin_angles + i]);
                    );
                }
                SYNC;
                MULTI(i, layer.size) {
                    #ifdef __CUDA_ARCH__
                    cache.activations[i] = unit_activation;
                    #else
                    cache.activations[i] = unit_activation[i];
                    #endif
                }
            }

            MULTI(j, layer.size) {
                function(layer.begin_params + j, cache.activations[j]);

                for(auto i = 0u; i < layer.lhs_connectivity; i++) {
                    // TODO: check if shared memory solution is faster
                    function(
                        layer.begin_params + layer.size + i * layer.size + j,
                        cache.activations[j] * (
                            layer_idx == 0 ?
                            spins[i] :
                            my_logcosh(
                                deep_angles[
                                    (this->layers[layer_idx - 1].begin_angles + i) % (
                                        this->layers[layer_idx - 1].size
                                    )
                                ]
                            )
                        )
                    );
                }
            }
        }
    }

    PsiDeep get_kernel() const {
        return *this;
    }

#endif // __CUDACC__

    HDINLINE
    float probability_s(const float log_psi_s_real) const {
        return exp(2.0f * (log(this->prefactor) + log_psi_s_real));
    }

    HDINLINE
    unsigned int get_num_spins() const {
        return this->N;
    }

    HDINLINE
    unsigned int get_num_params() const {
        return this->num_params;
    }

};

} // namespace kernel


class PsiDeep : public kernel::PsiDeep {
public:
    Array<complex_t> a_array;
    struct Layer {
        unsigned int        size;
        unsigned int        lhs_connectivity;

        Array<complex_t>    lhs_weights;
        Array<complex_t>    rhs_weights;
        Array<complex_t>    bases;
        Array<unsigned int> rhs_connections;
    };
    vector<Layer> layers;
    bool gpu;

    // vector<pair<int, int>> index_pair_list;

public:
    PsiDeep(const PsiDeep& other);

#ifdef __PYTHONCC__
    inline PsiDeep(
        const xt::pytensor<std::complex<float>, 1u>& a,
        const vector<xt::pytensor<std::complex<float>, 2u>>& lhs_weights_list,
        const vector<xt::pytensor<std::complex<float>, 1u>> bases_list,
        const float prefactor,
        const bool gpu
    ) : a_array(a, gpu), gpu(gpu) {
        this->N = a.shape()[0];
        this->prefactor = prefactor;
        this->num_layers = lhs_weights_list.size();

        Array<complex_t> rhs_weights_array(0, false);
        Array<unsigned int> rhs_connections_array(0, false);

        for(auto layer_idx = this->num_layers - 1; layer_idx >= 0; layer_idx--) {
            const auto& lhs_weights = lhs_weights_list[layer_idx];
            const auto& bases = bases_list[layer_idx];

            const unsigned int size = bases.size();
            const unsigned int lhs_connectivity = lhs_weights.shape()[0];

            Array<complex_t> lhs_weights_array(lhs_weights, gpu);
            Array<complex_t> bases_array(bases, gpu);

            const auto rhs_weights_and_connections = this->compile_rhs_weights_and_connections(
                layer_idx > 0 ? bases_list[layer_idx - 1].size() : this->N,
                size,
                lhs_connectivity,
                lhs_weights_array
            );

            this->layers.push_back({
                size,
                lhs_connectivity,
                move(lhs_weights_array),
                move(rhs_weights_array),
                move(bases_array),
                move(rhs_connections_array)
            });

            rhs_weights_array = move(rhs_weights_and_connections.first);
            rhs_connections_array = move(rhs_weights_and_connections.second);
        }

        this->init_kernel();
    }

    PsiDeep copy() const {
        return *this;
    }

    xt::pytensor<complex<float>, 1> get_params_py() const {
        auto result = xt::pytensor<complex<float>, 1>(
            std::array<long int, 1>({static_cast<long int>(this->num_params)})
        );
        this->get_params(result.data());

        return result;
    }

    void set_params_py(const xt::pytensor<complex<float>, 1>& new_params) {
        this->set_params(new_params.data());
    }

    unsigned int get_num_params_py() const {
        return this->get_num_params();
    }

#endif // __PYTHONCC__

    float norm_function(const ExactSummation& exact_summation) const;

    void get_params(complex<float>* result) const;
    void set_params(const complex<float>* new_params);

    void init_kernel();
    void update_kernel();

    pair<Array<complex_t>, Array<unsigned int>> compile_rhs_weights_and_connections(
        const unsigned int prev_size,
        const unsigned int size,
        const unsigned int lhs_connectivity,
        const Array<complex_t>& lhs_weights
    );
};

} // namespace rbm_on_gpu
