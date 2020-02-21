#ifdef ENABLE_PSI_PAIR


#include "quantum_state/PsiPair.hpp"

#include <complex>
#include <vector>
#include <random>
#include <cstring>
#include <algorithm>
#include <iterator>


namespace rbm_on_gpu {

PsiPair::PsiPair(const PsiPair& other)
    :
    alpha_array(other.alpha_array),
    beta_array(other.beta_array),
    psi_real(other.psi_real),
    psi_imag(other.psi_imag),
    free_quantum_axis(other.free_quantum_axis),
    gpu(other.gpu)
{
    this->prefactor = other.prefactor;
    this->update_kernel();
}

void PsiPair::update_kernel() {
    kernel::PsiPair::psi_real = this->psi_real.get_kernel();
    kernel::PsiPair::psi_imag = this->psi_imag.get_kernel();
}

Array<complex_t> PsiPair::get_params() const {
    const auto N = this->psi_real.N;
    const auto num_params = this->psi_real.num_params;
    Array<complex_t> result(num_params, false);

    const auto result_real = this->psi_real.get_params();
    const auto result_imag = this->psi_imag.get_params();

    for(auto i = 0u; i < N; i++) {
        result[i]= complex_t(this->alpha_array[i], 0.0);
        result[N + i] = complex_t(this->beta_array[i], 0.0);
    }

    for(auto k = 2u * N; k < num_params; k++) {
        result[k] = complex_t(result_real[k], result_imag[k]);
    }

    return result;
}


void PsiPair::set_params(const Array<complex_t>& new_params) {
    const auto N = this->psi_real.N;
    const auto num_params = this->psi_real.num_params;

    for(auto i = 0u; i < N; i++) {
        this->alpha_array[i] = get_real<double>(new_params[i]);
        this->beta_array[i] = get_real<double>(new_params[N + i]);
    }

    Array<double> real_params(num_params, false);
    Array<double> imag_params(num_params, false);

    for(auto k = 2u * N; k < this->psi_real.num_params; k++) {
        real_params[k] = new_params[k].real();
        imag_params[k] = new_params[k].imag();
    }

    this->psi_real.set_params(real_params);
    this->psi_imag.set_params(imag_params);

    this->update_kernel();
}

} // namespace rbm_on_gpu


#endif // ENABLE_PSI_PAIR
