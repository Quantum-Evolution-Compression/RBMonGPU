#include "quantum_state/PsiHistory.hpp"

namespace rbm_on_gpu {

PsiHistory::PsiHistory(const unsigned int num_steps, const unsigned int num_hidden_spins, const bool gpu)
    : gpu(gpu) {
    this->num_hidden_spins = num_hidden_spins;
    MALLOC(
        this->tanh_angles,
        sizeof(complex_t) * num_steps * num_hidden_spins,
        this->gpu
    );
}

PsiHistory::~PsiHistory() noexcept(false) {
    FREE(this->tanh_angles, this->gpu)
}

} // namespace rbm_on_gpu
