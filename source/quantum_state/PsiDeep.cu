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
    this->num_params = this->N;
    auto angle_idx = 0u;
    for(auto layer_idx = 0u; layer_idx < this->num_layers; layer_idx++) {
        const auto& layer = *next(this->layers.begin(), layer_idx);
        auto& kernel_layer = kernel::PsiDeep::layers[layer_idx];
        kernel_layer.size = layer.size;;
        kernel_layer.lhs_connectivity = layer.lhs_connectivity;

        kernel_layer.begin_params = this->num_params;
        kernel_layer.begin_angles = angle_idx;

        this->num_params += layer.size + layer.lhs_weights.size();
        angle_idx += layer.size;
    }
    for(auto layer_idx = 0u; layer_idx < this->num_layers; layer_idx++) {
        auto& layer = kernel::PsiDeep::layers[layer_idx];
        auto prev_layer = kernel::PsiDeep::layers + layer_idx - 1;
        auto next_layer = kernel::PsiDeep::layers + layer_idx + 1;

        const auto prev_size = (
            layer_idx == 0u ?
            this->N :
            prev_layer->size
        );

        layer.rhs_connectivity = (
            layer_idx + 1 < this->num_layers ?
            next_layer->size * next_layer->lhs_connectivity / layer.size :
            0u
        );
        layer.delta = (
            layer.size > prev_size ?
            1u :
            prev_size / layer.size
        );
    }

    this->update_kernel();
}


void PsiDeep::update_kernel() {
    this->a = this->a_array.data();

    for(auto layer_idx = 0u; layer_idx < this->num_layers; layer_idx++) {
        Layer& layer = *next(this->layers.begin(), layer_idx);
        auto& kernel_layer = kernel::PsiDeep::layers[layer_idx];

        kernel_layer.lhs_weights = layer.lhs_weights.data();
        kernel_layer.rhs_weights = layer.rhs_weights.data();
        kernel_layer.bases = layer.bases.data();
        kernel_layer.rhs_connections = layer.rhs_connections.data();
    }
}


pair<Array<complex_t>, Array<unsigned int>> PsiDeep::compile_rhs_weights_and_connections(
    const unsigned int prev_size,
    const unsigned int size,
    const unsigned int lhs_connectivity,
    const Array<complex_t>& lhs_weights
) {
    const auto rhs_connectivity = size * lhs_connectivity / prev_size;
    const auto delta = (
        size > prev_size ?
        1u :
        prev_size / size
    );

    Array<complex_t> rhs_weights(prev_size * rhs_connectivity, this->gpu);
    Array<unsigned int> rhs_connections(prev_size * rhs_connectivity, this->gpu);

    vector<unsigned int> lhs_num_connections;
    lhs_num_connections.assign(prev_size, 0u);

    for(auto j = 0u; j < size; j++) {
        for(auto i = 0u; i < lhs_connectivity; i++) {
            const auto lhs_idx = (delta * j + i) % prev_size;

            rhs_weights[lhs_idx * rhs_connectivity + lhs_num_connections[lhs_idx]] = lhs_weights[
                i * size + j
            ];
            rhs_connections[lhs_idx * rhs_connectivity + lhs_num_connections[lhs_idx]] = j;
            lhs_num_connections[lhs_idx]++;
        }
    }

    rhs_weights.update_device();
    rhs_connections.update_device();

    return {move(rhs_weights), move(rhs_connections)};
}


Array<complex_t> PsiDeep::get_params() const {
    Array<complex_t> result(this->num_params, false);

    auto it = result.begin();
    copy(this->a_array.begin(), this->a_array.end(), it);
    it += this->N;

    for(const auto& layer : this->layers) {
        copy(layer.bases.begin(), layer.bases.end(), it);
        it += layer.bases.size();
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

    for(auto layer_it = this->layers.begin(); layer_it != this->layers.end(); layer_it++) {
        auto& layer = *layer_it;

        copy(it, it + layer.bases.size(), layer.bases.begin());
        layer.bases.update_device();
        it += layer.size;

        copy(it, it + layer.lhs_weights.size(), layer.lhs_weights.begin());
        layer.lhs_weights.update_device();
        it += layer.lhs_weights.size();

        if(layer_it != this->layers.begin()) {
            prev(layer_it)->rhs_weights = this->compile_rhs_weights_and_connections(
                prev(layer_it)->size,
                layer.size,
                layer.lhs_connectivity,
                layer.lhs_weights
            ).first;
        }
    }

    this->update_kernel();
}

} // namespace rbm_on_gpu
