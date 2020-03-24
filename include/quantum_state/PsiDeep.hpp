#pragma once

#include "quantum_state/psi_functions.hpp"
#include "quantum_state/PsiDeepCache.hpp"
#include "quantum_state/PsiBase.hpp"

#include "Array.hpp"
#include "Spins.h"
#include "types.h"
#ifdef __CUDACC__
    #include "utils.kernel"
#endif
#include "cuda_complex.hpp"

#include <vector>
#include <list>
#include <complex>
#include <memory>
#include <cassert>
#include <utility>
#include <algorithm>

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pytensor.hpp"

    using namespace std::complex_literals;
#endif // __PYTHONCC__


// #define DIM 1


namespace rbm_on_gpu {

namespace kernel {

template<typename dtype>
struct PsiDeepT : public PsiBase {
    using Angles = rbm_on_gpu::PsiDeepAngles<dtype>;

    static constexpr unsigned int max_layers = 3u;
    static constexpr unsigned int max_deep_angles = 2u * MAX_SPINS;
    static constexpr unsigned int max_width = Angles::max_width;

    // TODO: Try to use stack-allocated arrays
    struct Layer {
        unsigned int  size;                 // number of units
        unsigned int  begin_angles;         // index of the first unit of this layer in a global list of angles
        unsigned int  begin_params;         // index of the first unit of this layer in a global list of parameters
        unsigned int  lhs_connectivity;     // number of connections to the lhs per unit
        unsigned int  rhs_connectivity;     // number of connections to the rhs per unit
        unsigned int* lhs_connections;      // connectivity matrix to the lhs: lhs-connectivity x size
        unsigned int* rhs_connections;      // connectivity matrix to the rhs: size x rhs-connectivity
        dtype*    lhs_weights;          // weight matrix to the lhs: lhs-connectivity x size, var.parameters
        dtype*    rhs_weights;          // weight matrix to the rhs: size x rhs-connectivity, var.parameters
        dtype*    biases;               // bias factors, var.parameters

        HDINLINE unsigned int lhs_connection(const unsigned int i, const unsigned int j) const {
            return this->lhs_connections[i * this->size + j];
        }
        HDINLINE unsigned int rhs_connection(const unsigned int i, const unsigned int j) const {
            return this->rhs_connections[i * this->rhs_connectivity + j];
        }
        HDINLINE dtype lhs_weight(const unsigned int i, const unsigned int j) const {
            return this->lhs_weights[i * this->size + j];
        }
        HDINLINE dtype rhs_weight(const unsigned int i, const unsigned int j) const {
            return this->rhs_weights[i * this->rhs_connectivity + j];
        }
    };

    Layer          layers[max_layers];
    dtype*         final_weights;
    unsigned int   num_final_weights;
    unsigned int   num_layers;
    unsigned int   width;                   // size of largest layer
    unsigned int   num_units;

    unsigned int   N_i;
    unsigned int   N_j;

    bool           translational_invariance;

public:

#ifdef __CUDACC__

    template<typename result_dtype>
    HDINLINE
    void forward_pass(
        result_dtype& result,  /* CAUTION: this parameter is assumed to be initialized */
        const Spins& spins,
        dtype* activations,
        dtype* deep_angles) const
    {
        #include "cuda_kernel_defines.h"

        MULTI(i, this->get_num_spins()) {
            activations[i] = spins[i];
        }
        SYNC;

        SHARED_MEM_LOOP_START(layer_idx, this->num_layers) {
            SYNC;
            const Layer& layer = this->layers[layer_idx];
            dtype REGISTER(activation, max_width);
            MULTI(j, layer.size) {
                REGISTER(activation, j) = dtype(0.0);
                // activations_out[j] = dtype(0.0);

                for(auto i = 0u; i < layer.lhs_connectivity; i++) {
                    REGISTER(activation, j) += (
                        layer.lhs_weight(i, j) *
                        activations[layer.lhs_connection(i, j)]
                    );
                }
                REGISTER(activation, j) += layer.biases[j];

                if(deep_angles != nullptr) {
                    deep_angles[layer.begin_angles + j] = REGISTER(activation, j);
                }
            }
            SYNC;
            MULTI(k, layer.size) {
                activations[k] = my_logcosh(REGISTER(activation, k));
            }
            SHARED_MEM_LOOP_END(layer_idx);
        }
        MULTI(j, this->num_final_weights) {
            generic_atomicAdd(&result, favour_real<result_dtype>(activations[j] * this->final_weights[j]));
        }
    }

    template<typename result_dtype>
    HDINLINE
    void log_psi_s_generic(result_dtype& result, const Spins& spins, Angles& cache) const {
        #include "cuda_kernel_defines.h"
        // CAUTION: 'result' has to be a shared variable.

        SINGLE {
            result = result_dtype(0.0);
        }

        SHARED Spins shifted_spins;

        SHARED_MEM_LOOP_START(shift, (this->translational_invariance ? this->N : 1u)) {
            SINGLE {
                shifted_spins = spins.rotate_left(shift, this->N);
            }
            SYNC;
            this->forward_pass(result, shifted_spins, cache.activations, nullptr);

            SHARED_MEM_LOOP_END(shift);
        }

        SINGLE {
            if(this->translational_invariance) {
                result *= 1.0 / this->N;
            }
        }

        // #if DIM == 2
        //     for(auto shift_i = 0u; shift_i < this->N_i; shift_i++) {
        //         for(auto shift_j = 0u; shift_j < this->N_j; shift_j++) {
        //             this->forward_pass(spins.shift_2d(shift_i, shift_j, this->N_i, this->N_j), cache.activations, nullptr);
        // #endif

    }

    HDINLINE
    void log_psi_s(dtype& result, const Spins& spins, Angles& cache) const {
        log_psi_s_generic(result, spins, cache);
    }

    HDINLINE
    void log_psi_s_real(double& result, const Spins& spins, Angles& cache) const {
        log_psi_s_generic(result, spins, cache);
    }

    HDINLINE
    dtype psi_s(const Spins& spins, Angles& cache) const {
        #include "cuda_kernel_defines.h"

        SHARED dtype log_psi;
        this->log_psi_s(log_psi, spins, cache);

        return exp(log(this->prefactor) + log_psi);
    }

    HDINLINE void flip_spin_of_jth_angle(
        const unsigned int j, const unsigned int position, const Spins& new_spins, Angles& angles
    ) const {
    }

    template<typename Function>
    HDINLINE
    void foreach_angle(const Spins& spins, Angles& cache, Function function) const {
        #include "cuda_kernel_defines.h"

        SHARED dtype deep_angles[max_deep_angles];
        SHARED complex_t log_psi;
        this->forward_pass(log_psi, spins, cache.activations, deep_angles);

        for(int layer_idx = int(this->num_layers) - 1; layer_idx >= 0; layer_idx--) {
            const Layer& layer = this->layers[layer_idx];

            MULTI(j, layer.size) {
                function(layer.begin_angles + j, deep_angles[layer.begin_angles + j]);
            }
        }
    }

    template<typename Function>
    HDINLINE
    void foreach_O_k(const Spins& spins, Angles& cache, Function function) const {
        #include "cuda_kernel_defines.h"

        SHARED dtype deep_angles[max_deep_angles];
        SHARED dtype log_psi;
        SINGLE {
            log_psi = dtype(0.0);
        }
        SYNC;

        this->forward_pass(log_psi, spins, cache.activations, deep_angles);

        for(int layer_idx = int(this->num_layers) - 1; layer_idx >= 0; layer_idx--) {
            const Layer& layer = this->layers[layer_idx];

            // calculate the activations of the layer.
            // here, these are the back-propagated derivatives.
            if(layer_idx == this->num_layers - 1) {
                MULTI(j, layer.size) {
                    cache.activations[j] = this->final_weights[j] * my_tanh(deep_angles[
                        layer.begin_angles + j
                    ]);
                }
            } else {
                // TODO: check if shared memory solution is faster (most likely not)
                dtype REGISTER(unit_activation, max_width);

                SYNC;
                MULTI(i, layer.size) {
                    REGISTER(unit_activation, i) = dtype(0.0);

                    for(auto j = 0u; j < layer.rhs_connectivity; j++) {
                        REGISTER(unit_activation, i) += (
                            layer.rhs_weight(i, j) * cache.activations[
                                layer.rhs_connection(i, j)
                            ]
                        );
                    }
                    REGISTER(unit_activation, i) *= (
                        my_tanh(deep_angles[layer.begin_angles + i])
                    );
                }
                SYNC;
                MULTI(j, layer.size) {
                    cache.activations[j] = REGISTER(unit_activation, j);
                }
            }
            MULTI(j, layer.size) {
                function(layer.begin_params + j, cache.activations[j]);

                for(auto i = 0u; i < layer.lhs_connectivity; i++) {
                    const auto lhs_unit_idx = layer.lhs_connection(i, j);
                    // TODO: check if shared memory solution is faster

                    function(
                        layer.begin_params + layer.size + i * layer.size + j,
                        cache.activations[j] * (
                            layer_idx == 0 ?
                            get_real<dtype>(spins[lhs_unit_idx]) :
                            my_logcosh(  // TODO: reverse for-loop such that logcosh is only evaluated once
                                deep_angles[
                                    this->layers[layer_idx - 1].begin_angles + lhs_unit_idx
                                ]
                            )
                        )
                    );
                }
            }
        }
        MULTI(j, this->num_final_weights) {
            function(
                this->num_params - this->num_final_weights + j,
                log_psi * my_logcosh(
                    deep_angles[this->layers[this->num_layers - 1].begin_angles + j]
                )
            );
        }
    }

    PsiDeepT get_kernel() const {
        return *this;
    }

#endif // __CUDACC__

    HDINLINE
    unsigned int get_width() const {
        return this->width;
    }


    HDINLINE unsigned int get_num_units() const {
        return this->num_units;
    }

    HDINLINE
    double probability_s(const double log_psi_s_real) const {
        return exp(2.0 * (log(this->prefactor) + log_psi_s_real));
    }

};

} // namespace kernel


template<typename dtype>
struct PsiDeepT : public kernel::PsiDeepT<dtype>, public PsiBase {
    struct Layer {
        unsigned int        size;
        unsigned int        lhs_connectivity;
        Array<unsigned int> lhs_connections;
        Array<unsigned int> rhs_connections;
        Array<dtype>        lhs_weights;
        Array<dtype>        rhs_weights;
        Array<dtype>        biases;
    };
    list<Layer> layers;
    Array<dtype> final_weights;

    PsiDeepT(const PsiDeepT& other);

#ifdef __PYTHONCC__

    inline PsiDeepT(
        const xt::pytensor<double, 1u>& alpha,
        const xt::pytensor<double, 1u>& beta,
        const vector<xt::pytensor<typename std_dtype<dtype>::type, 1u>> biases_list,
        const vector<xt::pytensor<unsigned int, 2u>>& lhs_connections_list,
        const vector<xt::pytensor<typename std_dtype<dtype>::type, 2u>>& lhs_weights_list,
        const xt::pytensor<typename std_dtype<dtype>::type, 1u>& final_weights,
        const double prefactor,
        const bool free_quantum_axis,
        const bool gpu
    ) : rbm_on_gpu::PsiBase(alpha, beta, free_quantum_axis, gpu), final_weights(final_weights, gpu) {
        this->N = alpha.shape()[0];
        this->prefactor = prefactor;
        this->num_layers = lhs_weights_list.size();
        this->width = this->N;
        this->num_units = 0u;
        this->translational_invariance = false;

        Array<unsigned int> rhs_connections_array(0, false);
        Array<dtype> rhs_weights_array(0, false);

        for(auto layer_idx = int(this->num_layers) - 1; layer_idx >= 0; layer_idx--) {
            const auto& lhs_connections = lhs_connections_list[layer_idx];
            const auto& lhs_weights = lhs_weights_list[layer_idx];
            const auto& biases = biases_list[layer_idx];

            const unsigned int size = biases.size();
            const unsigned int lhs_connectivity = lhs_connections.shape()[0];

            if(size > this->width) {
                this->width = size;
            }

            this->num_units += size;

            Array<unsigned int> lhs_connections_array(lhs_connections, gpu);
            Array<dtype> lhs_weights_array(lhs_weights, gpu);
            Array<dtype> biases_array(biases, gpu);

            const auto rhs_connections_and_weights = this->compile_rhs_connections_and_weights(
                layer_idx > 0 ? biases_list[layer_idx - 1].size() : this->N,
                size,
                lhs_connectivity,
                lhs_connections_array,
                lhs_weights_array
            );

            this->layers.push_front({
                size,
                lhs_connectivity,
                move(lhs_connections_array),
                move(rhs_connections_array),
                move(lhs_weights_array),
                move(rhs_weights_array),
                move(biases_array)
            });

            rhs_connections_array = move(rhs_connections_and_weights.first);
            rhs_weights_array = move(rhs_connections_and_weights.second);
        }

        this->init_kernel();

        // cout << "N: " << this->N << endl;
        // cout << "num_layers: " << this->num_layers << endl;
        // cout << "width: " << this->width << endl;
        // cout << "num_params: " << this->num_params << endl;
        // cout << "prefactor: " << this->prefactor << endl;
        // cout << endl;

        // for(auto layer_idx = int(this->num_layers) - 1; layer_idx >= 0; layer_idx--) {
        //     const auto& kernel_layer = kernel::PsiDeepT<dtype>::layers[layer_idx];
        //     const auto& layer = *next(this->layers.begin(), layer_idx);

        //     cout << "Layer: " << layer_idx << endl;
        //     cout << "size: " << kernel_layer.size << endl;
        //     cout << "lhs_connectivity: " << kernel_layer.lhs_connectivity << endl;
        //     cout << "rhs_connectivity: " << kernel_layer.rhs_connectivity << endl;
        //     cout << "begin_params: " << kernel_layer.begin_params << endl;
        //     cout << "begin_angles: " << kernel_layer.begin_angles << endl;
        //     cout << "lhs_weights.size: " << layer.lhs_weights.size() << endl;
        //     cout << "rhs_weights.size: " << layer.rhs_weights.size() << endl;
        //     cout << "biases.size: " << layer.biases.size() << endl;
        //     cout << "rhs_connections.size: " << layer.rhs_connections.size() << endl;
        //     cout << "lhs_connections: " << endl;
        //     for(auto i = 0u; i < layer.lhs_connectivity; i++) {
        //         for(auto j = 0u; j < layer.size; j++) {
        //             cout << layer.lhs_connections[i * layer.size + j] << ", ";
        //         }
        //         cout << endl;
        //     }
        //     cout << "rhs_connections: " << endl;
        //     for(auto i = 0u; i < layer.size; i++) {
        //         for(auto j = 0u; j < kernel_layer.rhs_connectivity; j++) {
        //             cout << layer.rhs_connections[i * kernel_layer.rhs_connectivity + j] << ", ";
        //         }
        //         cout << endl;
        //     }
        //     cout << endl;
        // }
    }

    PsiDeepT copy() const {
        return *this;
    }

    xt::pytensor<complex<double>, 1> O_k_vector_py(const Spins& spins) {
        return psi_O_k_vector_py(*this, spins);
    }

    inline vector<xt::pytensor<typename std_dtype<dtype>::type, 1u>> get_b() const {
        vector<xt::pytensor<typename std_dtype<dtype>::type, 1u>> result;

        for(const auto& layer : this->layers) {
            result.push_back(layer.biases.to_pytensor_1d());
        }

        return result;
    }

    inline vector<xt::pytensor<typename std_dtype<dtype>::type, 2u>> get_W() const {
        vector<xt::pytensor<typename std_dtype<dtype>::type, 2u>> result;

        for(const auto& layer : this->layers) {
            result.push_back(layer.lhs_weights.to_pytensor_2d(shape_t<2u>{
                (long int)layer.lhs_connectivity, (long int)layer.size
            }));
        }

        return result;
    }

    inline vector<xt::pytensor<unsigned int, 2>> get_connections() const {
        vector<xt::pytensor<unsigned int, 2>> result;

        for(const auto& layer : this->layers) {
            result.push_back(layer.lhs_connections.to_pytensor_2d(shape_t<2u>{
                (long int)layer.lhs_connectivity, (long int)layer.size
            }));
        }

        return result;
    }

#endif // __PYTHONCC__

    Array<dtype> get_params() const;
    void set_params(const Array<dtype>& new_params);

    void init_kernel();
    void update_kernel();

    pair<Array<unsigned int>, Array<dtype>> compile_rhs_connections_and_weights(
        const unsigned int prev_size,
        const unsigned int size,
        const unsigned int lhs_connectivity,
        const Array<unsigned int>& lhs_connections,
        const Array<dtype>& lhs_weights
    );
};


using PsiDeep = PsiDeepT<complex_t>;

} // namespace rbm_on_gpu
