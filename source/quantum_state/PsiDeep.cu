#include "quantum_state/PsiDeep.hpp"

#include <complex>
#include <vector>
#include <random>
#include <cstring>
#include <algorithm>
#include <iterator>


namespace rbm_on_gpu {


PsiDeep::PsiDeep(const PsiDeep& other)
    :
    a_array(other.a_array),
    alpha_array(other.alpha_array),
    layers(other.layers),
    gpu(other.gpu)
{
    this->N = other.N;
    this->prefactor = other.prefactor;
    this->num_layers = other.num_layers;
    this->width = other.width;
    this->num_units = other.num_units;

    this->init_kernel();
}


void PsiDeep::init_kernel() {
    this->num_params = 2 * this->N; // a and alpha
    auto angle_idx = 0u;
    for(auto layer_idx = 0u; layer_idx < this->num_layers; layer_idx++) {
        const auto& layer = *next(this->layers.begin(), layer_idx);
        auto& kernel_layer = kernel::PsiDeep::layers[layer_idx];
        kernel_layer.size = layer.size;
        kernel_layer.lhs_connectivity = layer.lhs_connectivity;

        kernel_layer.begin_params = this->num_params;
        kernel_layer.begin_angles = angle_idx;

        this->num_params += layer.size + layer.lhs_weights.size();
        angle_idx += layer.size;
    }
    for(auto layer_idx = 0u; layer_idx < this->num_layers; layer_idx++) {
        auto& layer = kernel::PsiDeep::layers[layer_idx];
        auto next_layer = kernel::PsiDeep::layers + layer_idx + 1;

        layer.rhs_connectivity = (
            layer_idx + 1 < this->num_layers ?
            next_layer->size * next_layer->lhs_connectivity / layer.size :
            0u
        );
    }

    this->update_kernel();
}


void PsiDeep::update_kernel() {
    this->a = this->a_array.data();

    for(auto layer_idx = 0u; layer_idx < this->num_layers; layer_idx++) {
        Layer& layer = *next(this->layers.begin(), layer_idx);
        auto& kernel_layer = kernel::PsiDeep::layers[layer_idx];

        kernel_layer.lhs_connections = layer.lhs_connections.data();
        kernel_layer.rhs_connections = layer.rhs_connections.data();
        kernel_layer.lhs_weights = layer.lhs_weights.data();
        kernel_layer.rhs_weights = layer.rhs_weights.data();
        kernel_layer.biases = layer.biases.data();
    }
}


pair<Array<unsigned int>, Array<complex_t>> PsiDeep::compile_rhs_connections_and_weights(
    const unsigned int prev_size,
    const unsigned int size,
    const unsigned int lhs_connectivity,
    const Array<unsigned int>& lhs_connections,
    const Array<complex_t>& lhs_weights
) {
    const auto rhs_connectivity = size * lhs_connectivity / prev_size;

    Array<unsigned int> rhs_connections(prev_size * rhs_connectivity, this->gpu);
    Array<complex_t> rhs_weights(prev_size * rhs_connectivity, this->gpu);

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

    rhs_connections.update_device();
    rhs_weights.update_device();

    return {move(rhs_connections), move(rhs_weights)};
}


Array<complex_t> PsiDeep::get_params() const {
    Array<complex_t> result(this->num_params, false);

    auto it = result.begin();
    copy(this->a_array.begin(), this->a_array.end(), it);
    it += this->N;
    copy(this->alpha_array.begin(), this->alpha_array.end(), it);
    it += this->N;

    for(const auto& layer : this->layers) {
        copy(layer.biases.begin(), layer.biases.end(), it);
        it += layer.biases.size();
        copy(layer.lhs_weights.begin(), layer.lhs_weights.end(), it);
        it += layer.lhs_weights.size();
    }

    return result;
}


void PsiDeep::set_params(const Array<complex_t>& new_params) {
    auto it = new_params.begin();

    copy(it, it + this->N, this->a_array.begin());
    this->a_array.update_device();
    it += this->N;
    for(auto i = 0u; i < this->N; i++) {
        this->alpha_array[i] = (*(it + i)).real();
    }
    it += this->N;

    for(auto layer_it = this->layers.begin(); layer_it != this->layers.end(); layer_it++) {
        auto& layer = *layer_it;

        copy(it, it + layer.biases.size(), layer.biases.begin());
        layer.biases.update_device();
        it += layer.size;

        copy(it, it + layer.lhs_weights.size(), layer.lhs_weights.begin());
        layer.lhs_weights.update_device();
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

} // namespace rbm_on_gpu
