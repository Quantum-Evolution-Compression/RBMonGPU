#include "spin_ensembles/ExactSummation.hpp"
#include "quantum_state/Psi.hpp"
#include "types.h"

#include <cassert>
#include <vector>
#include <algorithm>

using namespace std;


namespace rbm_on_gpu {

ExactSummation::ExactSummation(const unsigned int num_spins, const bool gpu)
    :
        gpu(gpu),
        num_spins(num_spins),
        allowed_spin_configurations_vec(nullptr)
    {
        this->num_spin_configurations = pow(2, num_spins);
        this->has_total_z_symmetry = false;
    }

void ExactSummation::set_total_z_symmetry(const int sector) {
    vector<Spins> spins_tmp;
    const auto hilbert_space_dim = pow(2, this->num_spins);

    for(auto spin_index = 0u; spin_index < hilbert_space_dim; spin_index++) {
        Spins spins = {(Spins::type)spin_index};

        if(spins.total_z(this->num_spins) == sector) {
            spins_tmp.push_back(spins);
        }
    }

    this->num_spin_configurations = spins_tmp.size();
    this->allowed_spin_configurations_vec = unique_ptr<Array<Spins>>(new Array<Spins>(spins_tmp.size(), this->gpu));
    this->allowed_spin_configurations_vec->assign(spins_tmp.begin(), spins_tmp.end());
    this->allowed_spin_configurations_vec->update_device();
    this->allowed_spin_configurations = this->allowed_spin_configurations_vec->data();

    this->has_total_z_symmetry = true;
}

} // namespace rbm_on_gpu
