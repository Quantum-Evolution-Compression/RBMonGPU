#pragma once

#ifdef __CUDACC__
#include "Spins.h"
#endif


#include <vector>
#include <list>
#include <unordered_map>
#include <complex>
#include <memory>
#include <cassert>
#include <utility>
#include <algorithm>
#include <iterator>
#include <string>
#include <fstream>
#include <iostream>

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

inline vector<int> binary_to_vector(const uint64_t spins, const unsigned int N) {
    vector<int> result(N);
    for(auto i = 0u; i < N; i++) {
        result[i] = (spins & (1u << i)) ? 1 : -1;
    }
    return result;
}

inline uint64_t vector_to_binary(const vector<int>& spins) {
    uint64_t result = 0u;
    for(auto i = 0u; i < spins.size(); i++) {
        result |= static_cast<uint64_t>(spins[i] == 1) << i;
    }
    return result;
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
    static constexpr unsigned int max_deep_angles = 2u * MAX_SPINS;
    static constexpr unsigned int max_width = 2u * MAX_SPINS;

#ifdef __CUDACC__
    struct Angles {
        Angles() = default;

        template<typename Psi_t>
        HDINLINE void init(const Psi_t& psi, const Angles& other) {
        }

        template<typename Psi_t>
        HDINLINE void init(const Psi_t& psi, const Spins& spins) {
        }
    };
#endif

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
    complex_std*   final_weights;
    unsigned int   num_final_weights;
    unsigned int   num_layers;

    unsigned int   num_params;
    double         prefactor;
    double         log_prefactor;

    vector<complex_std>*                    full_table_log_psi_ptr;
    unordered_map<uint64_t, complex_std>*   hash_table_log_psi_ptr;
    unsigned int                            max_hash_table_size;

    // using Angles = rbm_on_gpu::PsiDeepMinAngles;

public:

    inline
    void forward_pass(
        complex_std& result,
        const vector<int>& spins,
        complex_std* activations_in
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
        for(auto j = 0u; j < this->num_final_weights; j++) {
            result += activations_in[j] * this->final_weights[j];
        }
    }

    inline
    complex_std log_psi_s_raw(const vector<int>& spins_in) const {
        complex_std activations[max_width];

        complex_std result (0.0, 0.0);
        vector<int> spins(spins_in);

        for(auto shift = 0u; shift < this->N; shift++) {
            this->forward_pass(result, spins, activations);

            rotate(spins.begin(), spins.begin() + 1, spins.end());
        }
        result /= this->N;

        return result + this->log_prefactor;
    }

    inline
    complex_std log_psi_s(const vector<int>& spins_in) {
        if(this->full_table_log_psi_ptr != nullptr) {
            return this->full_table_log_psi_ptr->at(vector_to_binary(spins_in));
        }
        if(this->hash_table_log_psi_ptr != nullptr) {
            const auto spins = vector_to_binary(spins_in);
            const auto hit = this->hash_table_log_psi_ptr->find(spins);
            if(hit != this->hash_table_log_psi_ptr->end()) {
                return hit->second;
            }

            const auto result = this->log_psi_s_raw(spins_in);
            this->hash_table_log_psi_ptr->emplace(spins, result);
            if(this->hash_table_log_psi_ptr->size() > this->max_hash_table_size) {
                this->hash_table_log_psi_ptr->clear();
            }
            return result;
        }

        return this->log_psi_s_raw(spins_in);
    }

#ifdef __CUDACC__
    HDINLINE void log_psi_s(complex_t& result, const Spins& spins, const Angles& angles) const {
        #ifndef __CUDA_ARCH__
        std::vector<int> spins_vec(this->N);
        for(auto i = 0u; i < this->N; i++) {
            spins_vec[i] = spins[i];
        }
        result = complex_t(this->log_psi_s_raw(spins_vec)) - this->log_prefactor;

        #endif
    }

    HDINLINE double probability_s(const double log_psi_s_real) const {
        return exp(2.0 * (this->log_prefactor + log_psi_s_real));
    }

    HDINLINE unsigned int get_num_spins() const {
        return this->N;
    }

    HDINLINE unsigned int get_width() const {
        return this->N;
    }


#endif

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
    vector<complex_std> final_weights;

    vector<complex_std>                    full_table_log_psi;
    unordered_map<uint64_t, complex_std>   hash_table_log_psi;

    bool gpu;

    inline PsiDeepMin(const string data_file_name) : gpu(false) {
        ifstream infile(data_file_name);
        if(infile.is_open()) {
            cout << data_file_name << " was successfully loaded." << endl;
        } else {
            cout << data_file_name << " was NOT loaded!" << endl;
        }

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
        this->full_table_log_psi_ptr = nullptr;
        this->hash_table_log_psi_ptr = nullptr;
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
        copy(it, it + this->num_final_weights, back_inserter(this->final_weights));
        it += this->num_final_weights;

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
        this->num_final_weights = this->layers.back().size;
        this->num_params += this->num_final_weights;
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
        kernel::PsiDeepMin::final_weights = PsiDeepMin::final_weights.data();
    }

    inline void enable_full_table() {
        this->full_table_log_psi.resize(1u << this->N);

        for(auto spins = 0u; spins < (1u << this->N); spins++) {
            this->full_table_log_psi[spins] = this->log_psi_s_raw(binary_to_vector(spins, this->N));
        }

        this->full_table_log_psi_ptr = &(this->full_table_log_psi);
    }

    inline void enable_hash_table(const unsigned int max_size) {
        this->max_hash_table_size = max_size;
        this->hash_table_log_psi_ptr = &(this->hash_table_log_psi);
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
