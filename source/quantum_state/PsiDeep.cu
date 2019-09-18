#include "quantum_state/PsiDeep.hpp"

#include <complex>
#include <vector>
#include <random>
#include <cstring>
#include <algorithm>


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
        layer.delta = min(
            layer.lhs_connectivity, prev_size - layer.lhs_connectivity
        ) % prev_size;
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
    const auto delta = min(
        lhs_connectivity, prev_size - lhs_connectivity
    );

    Array<complex_t> rhs_weights(prev_size * rhs_connectivity, this->gpu);
    Array<unsigned int> rhs_connections(prev_size * rhs_connectivity, this->gpu);

    vector<unsigned int> lhs_num_connections;
    lhs_num_connections.assign(prev_size, 0u);

    for(auto j = 0u; j < size; j++) {
        for(auto i = 0u; i < rhs_connectivity; i++) {
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

} // namespace rbm_on_gpu
