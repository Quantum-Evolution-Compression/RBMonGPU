#pragma once

#include "quantum_state/psi_functions.hpp"
#include "quantum_state/PsiDeepCache.hpp"
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


#define TRANSLATIONAL_INVARIANCE
#define DIM 1


namespace rbm_on_gpu {

namespace kernel {

template<typename dtype>
class PsiDeepT {
public:
    using Angles = rbm_on_gpu::PsiDeepAngles<dtype>;

    static constexpr unsigned int max_layers = 3u;
    static constexpr unsigned int max_deep_angles = max_layers * 2 * MAX_SPINS;

    // TODO: Try to use stack-allocated arrays
    struct Layer {
        unsigned int  size;                 // number of units
        unsigned int  begin_angles;         // index of the first unit of this layer in a global list of angles
        unsigned int  begin_params;         // index of the first unit of this layer in a global list of parameters
        unsigned int  lhs_connectivity;     // number of connections to the lhs per unit
        unsigned int  rhs_connectivity;     // number of connections to the rhs per unit
        unsigned int* lhs_connections;      // connectivity matrix to the lhs: lhs-connectivity x size
        unsigned int* rhs_connections;      // connectivity matrix to the rhs: size x rhs-connectivity
        dtype*    lhs_weights;          // weight matrix to the lhs: lhs-connectivity x size
        dtype*    rhs_weights;          // weight matrix to the rhs: size x rhs-connectivity
        dtype*    biases;               // bias factors

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

    unsigned int   N;
    Layer          layers[max_layers];
    unsigned int   num_layers;
    unsigned int   width;                   // size of largest layer
    unsigned int   num_units;

    unsigned int   N_i;
    unsigned int   N_j;

    unsigned int   num_params;
    unsigned int   O_k_length;
    double         prefactor;

public:

#ifdef __CUDACC__

    HDINLINE
    void forward_pass(
        const Spins& spins,
        dtype* activations_in, /* once this functions has finished, this holds the *output*-activations of the last layer */
        dtype* deep_angles) const
    {
        #include "cuda_kernel_defines.h"

        SHARED dtype activations_out[Angles::max_width];

        MULTI(i, this->get_num_spins()) {
            activations_in[i] = spins[i];
        }

        for(auto layer_idx = 0u; layer_idx < this->num_layers; layer_idx++) {
            SYNC;
            const Layer& layer = this->layers[layer_idx];
            MULTI(j, layer.size) {
                activations_out[j] = dtype(0.0);

                for(auto i = 0u; i < layer.lhs_connectivity; i++) {
                    activations_out[j] += (
                        layer.lhs_weight(i, j) *
                        activations_in[layer.lhs_connection(i, j)]
                    );
                }
                activations_out[j] += layer.biases[j];

                if(deep_angles != nullptr) {
                    deep_angles[layer.begin_angles + j] = activations_out[j];
                }
            }
            SYNC;
            MULTI(k, layer.size) {
                activations_in[k] = my_logcosh(activations_out[k]);
            }
        }
    }

    HDINLINE
    void log_psi_s(dtype& result, const Spins& spins, Angles& cache) const {
        // CAUTION: 'result' has to be a shared variable.

        SINGLE {
            result = dtype(0.0);
        }

        #ifdef TRANSLATIONAL_INVARIANCE

        #if DIM == 1
        for(auto shift = 0u; shift < this->N; shift++) {
        this->forward_pass(spins.rotate_left(shift, this->N), cache.activations, nullptr);
        #endif
        #if DIM == 2
        for(auto shift_i = 0u; shift_i < this->N_i; shift_i++) {
            for(auto shift_j = 0u; shift_j < this->N_j; shift_j++) {
                this->forward_pass(spins.shift_2d(shift_i, shift_j, this->N_i, this->N_j), cache.activations, nullptr);
        #endif

        #else

        this->forward_pass(spins, cache.activations, nullptr);

        #endif

        MULTI(j, this->layers[this->num_layers - 1u].size) {
            generic_atomicAdd(&result, cache.activations[j]);
        }


        #ifdef TRANSLATIONAL_INVARIANCE

        #if DIM == 1
        }
        #endif
        #if DIM == 2
        }}
        #endif

        result *= 1.0 / this->N;

        #endif
    }

    HDINLINE
    void log_psi_s_real(double& result, const Spins& spins, Angles& cache) const {
        // CAUTION: 'result' has to be a shared variable.

        SINGLE {
            result = 0.0;
        }

        #ifdef TRANSLATIONAL_INVARIANCE

        #if DIM == 1
        for(auto shift = 0u; shift < this->N; shift++) {
        this->forward_pass(spins.rotate_left(shift, this->N), cache.activations, nullptr);
        #endif
        #if DIM == 2
        for(auto shift_i = 0u; shift_i < this->N_i; shift_i++) {
            for(auto shift_j = 0u; shift_j < this->N_j; shift_j++) {
                this->forward_pass(spins.shift_2d(shift_i, shift_j, this->N_i, this->N_j), cache.activations, nullptr);
        #endif

        #else

        this->forward_pass(spins, cache.activations, nullptr);

        #endif

        MULTI(j, this->layers[this->num_layers - 1u].size) {
            generic_atomicAdd(&result, cache.activations[j].real());
        }


        #ifdef TRANSLATIONAL_INVARIANCE

        #if DIM == 1
        }
        #endif
        #if DIM == 2
        }}
        #endif

        result *= 1.0 / this->N;

        #endif
    }

    HDINLINE void flip_spin_of_jth_angle(
        const unsigned int j, const unsigned int position, const Spins& new_spins, Angles& cache
    ) const {
    }

    HDINLINE
    dtype psi_s(const Spins& spins, Angles& cache) const {
        #include "cuda_kernel_defines.h"

        SHARED dtype log_psi;
        this->log_psi_s(log_psi, spins, cache);

        return exp(log(this->prefactor) + log_psi);
    }

    template<typename Function>
    HDINLINE
    void foreach_angle(const Spins& spins, Angles& cache, Function function) const {
        #include "cuda_kernel_defines.h"

        SHARED dtype deep_angles[max_deep_angles];
        this->forward_pass(spins, cache.activations, deep_angles);

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

        this->forward_pass(spins, cache.activations, deep_angles);

        for(int layer_idx = int(this->num_layers) - 1; layer_idx >= 0; layer_idx--) {
            const Layer& layer = this->layers[layer_idx];

            // calculate the activations of the layer.
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
                dtype unit_activation(0.0);
                #else
                dtype unit_activation[Angles::max_width];
                #endif

                SYNC;
                MULTI(i, layer.size) {
                    #ifndef __CUDA_ARCH__
                    unit_activation[i] = dtype(0.0);
                    #endif

                    for(auto j = 0u; j < layer.rhs_connectivity; j++) {
                        #ifdef __CUDA_ARCH__
                        unit_activation +=
                        #else
                        unit_activation[i] +=
                        #endif
                        (
                            layer.rhs_weight(i, j) * cache.activations[
                                layer.rhs_connection(i, j)
                            ]
                        );
                    }
                    #ifdef __CUDA_ARCH__
                    unit_activation *=
                    #else
                    unit_activation[i] *=
                    #endif
                    (
                        my_tanh(deep_angles[layer.begin_angles + i])
                    );
                }
                SYNC;
                MULTI(j, layer.size) {
                    #ifdef __CUDA_ARCH__
                    cache.activations[j] = unit_activation;
                    #else
                    cache.activations[j] = unit_activation[j];
                    #endif
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
                            dtype(spins[lhs_unit_idx], 0.0) :
                            my_logcosh(
                                deep_angles[
                                    this->layers[layer_idx - 1].begin_angles + lhs_unit_idx
                                ]
                            )
                        )
                    );
                }
            }
        }
    }

    PsiDeepT get_kernel() const {
        return *this;
    }

#endif // __CUDACC__

    HDINLINE
    double probability_s(const double log_psi_s_real) const {
        return exp(2.0 * (log(this->prefactor) + log_psi_s_real));
    }

    HDINLINE
    unsigned int get_num_spins() const {
        return this->N;
    }

    HDINLINE
    unsigned int get_num_params() const {
        return this->num_params;
    }

    HDINLINE
    unsigned int get_width() const {
        return this->width;
    }

    HDINLINE
    unsigned int get_num_angles() const {
        return this->layers[0].size;
    }

    HDINLINE
    unsigned int get_num_units() const {
        return this->num_units;
    }

    HDINLINE
    unsigned int get_O_k_length() const {
        return this->O_k_length;
    }
};

} // namespace kernel


template<typename dtype>
class PsiDeepT : public kernel::PsiDeepT<dtype> {
public:
    Array<double> alpha_array;
    Array<double> beta_array;
    const bool    free_quantum_axis;

    struct Layer {
        unsigned int        size;
        unsigned int        lhs_connectivity;
        Array<unsigned int> lhs_connections;
        Array<unsigned int> rhs_connections;
        Array<dtype>    lhs_weights;
        Array<dtype>    rhs_weights;
        Array<dtype>    biases;
    };
    list<Layer> layers;

    bool gpu;

public:
    PsiDeepT(const PsiDeepT& other);

#ifdef __PYTHONCC__

    inline PsiDeepT(
        const xt::pytensor<double, 1u>& alpha,
        const xt::pytensor<double, 1u>& beta,
        const vector<xt::pytensor<typename std_dtype<dtype>::type, 1u>> biases_list,
        const vector<xt::pytensor<unsigned int, 2u>>& lhs_connections_list,
        const vector<xt::pytensor<typename std_dtype<dtype>::type, 2u>>& lhs_weights_list,
        const double prefactor,
        const bool free_quantum_axis,
        const bool gpu
    ) : alpha_array(alpha, false), beta_array(beta, false), free_quantum_axis(free_quantum_axis), gpu(gpu) {
        this->N = alpha.shape()[0];
        this->prefactor = prefactor;
        this->num_layers = lhs_weights_list.size();
        this->width = this->N;
        this->num_units = 0u;

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
        //     const auto& kernel_layer = kernel::PsiDeep::layers[layer_idx];
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

    inline vector<xt::pytensor<complex<double>, 1>> get_b() const {
        vector<xt::pytensor<complex<double>, 1>> result;

        for(const auto& layer : this->layers) {
            result.push_back(layer.biases.to_pytensor<1u>());
        }

        return result;
    }

    inline vector<xt::pytensor<complex<double>, 2>> get_W() const {
        vector<xt::pytensor<complex<double>, 2>> result;

        for(const auto& layer : this->layers) {
            result.push_back(layer.lhs_weights.to_pytensor<2u>(shape_t<2u>{
                (long int)layer.lhs_connectivity, (long int)layer.size
            }));
        }

        return result;
    }

    inline vector<xt::pytensor<unsigned int, 2>> get_connections() const {
        vector<xt::pytensor<unsigned int, 2>> result;

        for(const auto& layer : this->layers) {
            result.push_back(layer.lhs_connections.to_pytensor<2u>(shape_t<2u>{
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

private:
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
