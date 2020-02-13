#pragma once

#include <vector>
#include <list>
#include <complex>
#include <memory>
#include <cassert>
#include <utility>
#include <algorithm>
#include <string>
#include <fstream>

#ifndef MAX_SPINS
#define MAX_SPINS 64
#endif

using namespace std;
using complex_std = std::complex<double>;

inline complex_std read_complex(ifstream& infile) {
    double value_re, value_im;
    infile >> value_re >> value_im;
    return complex_std(value_re, value_im);
}


namespace rbm_on_gpu {

namespace kernel {

inline complex_std activation_function(const complex_std z) {
    const auto sign = z.real() > 0.0 ? 1.0 : -1.0;

    return sign * 0.9003320053750442 * z + (
        5.49914721954 - sign * 2.16564366435 * z
    ) / (
        9.19376335670885 + z * (sign * 10.2180213465 + z * (7.771429504240965 + z * (sign * 3.746646023906276 + z)))
    ) - 0.598139;
}


class PsiDeepMin {
public:
    static constexpr unsigned int max_layers = 3u;
    static constexpr unsigned int max_width = 2 * MAX_SPINS;
    static constexpr unsigned int max_deep_angles = max_layers * MAX_SPINS;

    // TODO: Try to use stack-allocated arrays
    struct Layer {
        unsigned int  size;                 // number of units
        unsigned int  begin_params;         // index of the first unit of this layer in a global list of parameters
        unsigned int  lhs_connectivity;     // number of connections to the lhs per unit
        unsigned int  rhs_connectivity;     // number of connections to the rhs per unit
        unsigned int* lhs_connections;      // connectivity matrix to the lhs: lhs-connectivity x size
        unsigned int* rhs_connections;      // connectivity matrix to the rhs: size x rhs-connectivity
        complex_std*    lhs_weights;          // weight matrix to the lhs: lhs-connectivity x size
        complex_std*    rhs_weights;          // weight matrix to the rhs: size x rhs-connectivity
        complex_std*    biases;               // bias factors

        inline unsigned int lhs_connection(const unsigned int i, const unsigned int j) const {
            return this->lhs_connections[i * this->size + j];
        }
        inline unsigned int rhs_connection(const unsigned int i, const unsigned int j) const {
            return this->rhs_connections[i * this->rhs_connectivity + j];
        }
        inline complex_std lhs_weight(const unsigned int i, const unsigned int j) const {
            return this->lhs_weights[i * this->size + j];
        }
        inline complex_std rhs_weight(const unsigned int i, const unsigned int j) const {
            return this->rhs_weights[i * this->rhs_connectivity + j];
        }
    };

    unsigned int   N;
    Layer          layers[max_layers];
    unsigned int   num_layers;

    unsigned int   num_params;
    double         prefactor;
    double         log_prefactor;

    // using Angles = rbm_on_gpu::PsiDeepMinAngles;

public:

    inline
    void forward_pass(
        const vector<int>& spins,
        complex_std* activations_in /* once this functions has finished, this holds the *output*-activations of the last layer */
    ) const
    {
        complex_std activations_out[max_width];

        for(auto i = 0u; i < this->N; i++) {
            activations_in[i] = spins[i];
        }

        for(auto layer_idx = 0u; layer_idx < this->num_layers; layer_idx++) {

            const Layer& layer = this->layers[layer_idx];
            for(auto j = 0u; j < layer.size; j++) {
                activations_out[j] = complex_std(0.0, 0.0);

                for(auto i = 0u; i < layer.lhs_connectivity; i++) {
                    activations_out[j] += (
                        layer.lhs_weight(i, j) *
                        activations_in[layer.lhs_connection(i, j)]
                    );
                }
                activations_out[j] += layer.biases[j];
            }

            for(auto k = 0u; k < layer.size; k++) {
                activations_in[k] = activation_function(activations_out[k]);
            }
        }
    }

    inline
    complex_std log_psi_s(const vector<int>& spins_in) const {
        complex_std activations[max_width];

        complex_std result (0.0, 0.0);
        vector<int> spins(spins_in);

        for(auto shift = 0u; shift < this->N; shift++) {
            this->forward_pass(spins, activations);

            for(auto j = 0u; j < this->layers[this->num_layers - 1u].size; j++) {
                result += activations[j];
            }

            rotate(spins.begin(), spins.begin() + 1, spins.end());
        }
        result /= this->N;

        return result + this->log_prefactor;
    }

    PsiDeepMin& get_kernel() {
        return *this;
    }

    const PsiDeepMin& get_kernel() const {
        return *this;
    }
};

} // namespace kernel


class PsiDeepMin : public kernel::PsiDeepMin {
public:

    struct Layer {
        unsigned int        size;
        unsigned int        lhs_connectivity;
        vector<unsigned int> lhs_connections;
        vector<unsigned int> rhs_connections;
        vector<complex_std>    lhs_weights;
        vector<complex_std>    rhs_weights;
        vector<complex_std>    biases;
    };
    list<Layer> layers;

    bool gpu;

    inline PsiDeepMin(const string data_file_name) : gpu(false) {
        ifstream infile(data_file_name);

        // read fixed params

        infile >> this->N;
        infile >> this->num_layers;
        infile >> this->log_prefactor;
        this->prefactor = exp(this->log_prefactor);

        vector<unsigned int> layer_sizes;
        for(auto i = 0u; i < this->num_layers; i++) {
            unsigned int size;
            infile >> size;

            layer_sizes.push_back(size);
        }

        vector<unsigned int> rhs_connections_array(0);
        vector<complex_std> rhs_weights_array(0);

        for(auto layer_idx = int(this->num_layers) - 1; layer_idx >= 0; layer_idx--) {
            const auto size = layer_sizes[layer_idx];

            unsigned int lhs_connectivity;
            infile >> lhs_connectivity;

            vector<unsigned int> lhs_connections_array(lhs_connectivity * size);
            for(auto i = 0u; i < lhs_connectivity; i++) {
                for(auto j = 0u; j < size; j++) {
                    infile >> lhs_connections_array[i * size + j];
                }
            }
            vector<complex_std> lhs_weights_array(lhs_connectivity * size);
            vector<complex_std> biases_array(size);

            const auto rhs_connections_and_weights = this->compile_rhs_connections_and_weights(
                layer_idx > 0 ? layer_sizes[layer_idx - 1] : this->N,
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

        // read dynamical params

        vector<complex_std> dynamical_params(this->num_params);
        for(auto i = 0u; i < this->num_params; i++) {
            dynamical_params[i] = read_complex(infile);
        }
        infile.close();

        this->set_params(dynamical_params);
    }

    inline void set_params(const vector<complex_std>& new_params) {
        auto it = new_params.begin();

        for(auto layer_it = this->layers.begin(); layer_it != this->layers.end(); layer_it++) {
            auto& layer = *layer_it;

            copy(it, it + layer.biases.size(), layer.biases.begin());
            it += layer.size;

            copy(it, it + layer.lhs_weights.size(), layer.lhs_weights.begin());
            it += layer.lhs_weights.size();

            if(layer_it != this->layers.begin()) {
                prev(layer_it)->rhs_weights = this->compile_rhs_connections_and_weights(
                    prev(layer_it)->size,
                    layer.size,
                    layer.lhs_connectivity,
                    layer.lhs_connections,
                    layer.lhs_weights
                ).second;
            }
        }

        this->update_kernel();
    }

    inline void init_kernel() {
        this->num_params = 0u;
        for(auto layer_idx = 0u; layer_idx < this->num_layers; layer_idx++) {
            const auto& layer = *next(this->layers.begin(), layer_idx);
            auto& kernel_layer = kernel::PsiDeepMin::layers[layer_idx];
            kernel_layer.size = layer.size;
            kernel_layer.lhs_connectivity = layer.lhs_connectivity;

            kernel_layer.begin_params = this->num_params;

            this->num_params += layer.size + layer.lhs_weights.size();
        }
        for(auto layer_idx = 0u; layer_idx < this->num_layers; layer_idx++) {
            auto& layer = kernel::PsiDeepMin::layers[layer_idx];
            auto next_layer = kernel::PsiDeepMin::layers + layer_idx + 1;

            layer.rhs_connectivity = (
                layer_idx + 1 < this->num_layers ?
                next_layer->size * next_layer->lhs_connectivity / layer.size :
                0u
            );
        }
        this->update_kernel();
    }

    inline void update_kernel() {
        for(auto layer_idx = 0u; layer_idx < this->num_layers; layer_idx++) {
            Layer& layer = *next(this->layers.begin(), layer_idx);
            auto& kernel_layer = kernel::PsiDeepMin::layers[layer_idx];

            kernel_layer.lhs_connections = layer.lhs_connections.data();
            kernel_layer.rhs_connections = layer.rhs_connections.data();
            kernel_layer.lhs_weights = layer.lhs_weights.data();
            kernel_layer.rhs_weights = layer.rhs_weights.data();
            kernel_layer.biases = layer.biases.data();
        }
    }

private:
    inline pair<vector<unsigned int>, vector<complex_std>> compile_rhs_connections_and_weights(
        const unsigned int prev_size,
        const unsigned int size,
        const unsigned int lhs_connectivity,
        const vector<unsigned int>& lhs_connections,
        const vector<complex_std>& lhs_weights
    ) {
        const auto rhs_connectivity = size * lhs_connectivity / prev_size;

        vector<unsigned int> rhs_connections(prev_size * rhs_connectivity);
        vector<complex_std> rhs_weights(prev_size * rhs_connectivity);

        vector<unsigned int> lhs_num_connections;
        lhs_num_connections.assign(prev_size, 0u);

        for(auto j = 0u; j < size; j++) {
            for(auto i = 0u; i < lhs_connectivity; i++) {
                const auto lhs_idx = lhs_connections[i * size + j];

                rhs_connections[lhs_idx * rhs_connectivity + lhs_num_connections[lhs_idx]] = j;
                rhs_weights[lhs_idx * rhs_connectivity + lhs_num_connections[lhs_idx]] = lhs_weights[
                    i * size + j
                ];
                lhs_num_connections[lhs_idx]++;
            }
        }

        return {move(rhs_connections), move(rhs_weights)};
    }
};


} // namespace rbm_on_gpu
