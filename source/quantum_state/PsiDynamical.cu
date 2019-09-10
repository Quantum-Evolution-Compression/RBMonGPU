#include "quantum_state/PsiDynamical.hpp"
#include "network_functions/PsiVector.hpp"
#include "network_functions/PsiNorm.hpp"
#include "network_functions/PsiOkVector.hpp"

#include <algorithm>
#include <utility>


namespace rbm_on_gpu {

using clist = PsiDynamical::clist;


PsiDynamical::PsiDynamical(const clist& spin_weights, const bool gpu)
    : spin_weights(spin_weights), gpu(gpu)
{
    this->init(spin_weights.data(), spin_weights.size(), false);
}

PsiDynamical::PsiDynamical(const PsiDynamical& other)
    :   spin_weights(other.spin_weights),
        links(other.links),
        gpu(other.gpu)
{
    this->init(reinterpret_cast<complex<double>*>(other.a), other.N, other.gpu, other.prefactor);
    this->update(true);
}

PsiDynamical::~PsiDynamical() noexcept(false) {
    FREE(this->a, this->gpu);
    this->clear_hidden_spins();
}

void PsiDynamical::init(const complex<double>* a, const unsigned int N, const bool a_on_gpu, const double prefactor) {
    this->N = N;
    this->prefactor = prefactor;

    MALLOC(this->a, sizeof(complex<double>) * this->N, this->gpu);
    MEMCPY(this->a, a, sizeof(complex<double>) * this->N, this->gpu, a_on_gpu);
    this->b = nullptr;
    this->W = nullptr;
    this->n = nullptr;

    this->spin_offset_list = nullptr;
    this->W_offset_list = nullptr;
    this->param_offset_list = nullptr;
    this->string_length_list = nullptr;
    this->hidden_spin_type_list = nullptr;

    for(int i = 0; i < int(N); i++) {
        this->index_pair_list.push_back(make_pair(i, -1));
    }
}

void PsiDynamical::clear_hidden_spins() {
    FREE(this->b, this->gpu);
    FREE(this->W, this->gpu);
    FREE(this->spin_offset_list, this->gpu);
    FREE(this->W_offset_list, this->gpu);
    FREE(this->param_offset_list, this->gpu);
    FREE(this->string_length_list, this->gpu);
    FREE(this->hidden_spin_type_list, this->gpu);
}

unsigned int PsiDynamical::sizeof_W() const {
    auto result = 0u;
    for(const auto& link : this->links) {
        result += link.weights.size();
    }

    return result;
}

void PsiDynamical::as_vector(complex<double>* result) const {
    psi_vector(result, *this);
}

double PsiDynamical::norm_function(const ExactSummation& exact_summation) const {
    return psi_norm(*this, exact_summation);
}

void PsiDynamical::O_k_vector(complex<double>* result, const Spins& spins) const {
    psi_O_k_vector(result, *this, spins);
}

void PsiDynamical::dense_W(complex<double>* result) const {
    memset(result, 0, sizeof(complex_t) * this->N * this->M);

    auto j = 0u;
    for(const auto& link : this->links) {
        for(auto i = link.first_spin; i < link.first_spin + link.weights.size(); i++) {
            result[(i % this->N) * this->M + j] = link.weights[i - link.first_spin];
        }

        j++;
    }
}

void PsiDynamical::add_hidden_spin(
    const unsigned int first_spin, const clist& link_weights, const complex<double>& hidden_spin_weight, const int hidden_spin_type
) {
    this->links.push_back(Link{first_spin, link_weights, hidden_spin_weight, hidden_spin_type});
}

void PsiDynamical::update(bool resize) {
    this->M = this->links.size();
    const auto sizeof_W = this->sizeof_W();

    if(resize) {
        this->clear_hidden_spins();

        MALLOC(this->b, sizeof(complex_t) * this->M, this->gpu);
        MALLOC(this->W, sizeof(complex_t) * sizeof_W, this->gpu);
        MALLOC(this->spin_offset_list, sizeof(unsigned int) * this->M, this->gpu);
        MALLOC(this->W_offset_list, sizeof(unsigned int) * this->M, this->gpu);
        MALLOC(this->param_offset_list, sizeof(unsigned int) * this->M, this->gpu);
        MALLOC(this->string_length_list, sizeof(unsigned int) * this->M, this->gpu);
        MALLOC(this->hidden_spin_type_list, sizeof(unsigned int) * this->M, this->gpu);
    }

    clist b_host;
    clist W_host;

    vector<unsigned int> spin_offset_list_host;
    vector<unsigned int> W_offset_list_host;
    vector<unsigned int> param_offset_list_host;
    vector<unsigned int> string_length_list_host;
    vector<unsigned int> hidden_spin_type_host;

    b_host.reserve(this->M);
    W_host.reserve(sizeof_W);
    if(resize) {
        spin_offset_list_host.reserve(this->M);
        W_offset_list_host.reserve(this->M);
        param_offset_list_host.reserve(this->M);
        string_length_list_host.reserve(this->M);
        hidden_spin_type_host.reserve(this->M);

        this->num_active_params = N + M;

        this->index_pair_list.resize(N + M);
        for(int j = 0; j < int(M); j++) {
            this->index_pair_list[N + j] = make_pair(-1, j);
        }
    }

    auto W_offset = 0u;
    auto param_offset = this->N + this->M;

    int j = 0;
    for(const auto& link : this->links) {
        b_host.push_back(link.hidden_spin_weight);
        copy(link.weights.begin(), link.weights.end(), back_inserter(W_host));

        if(resize) {
            spin_offset_list_host.push_back(link.first_spin);
            W_offset_list_host.push_back(W_offset);
            param_offset_list_host.push_back(param_offset);
            string_length_list_host.push_back(link.weights.size());
            hidden_spin_type_host.push_back(link.hidden_spin_type);

            W_offset += link.weights.size();
            param_offset += link.weights.size();
            this->num_active_params += link.weights.size();

            for(int i = link.first_spin; i < link.first_spin + link.weights.size(); i++) {
                this->index_pair_list.push_back(make_pair(i % N, j));
            }
        }

        j++;
    }

    if(resize) {
        this->num_params = this->num_active_params;
    }

    MEMCPY(this->a, this->spin_weights.data(), sizeof(complex_t) * this->N, this->gpu, false);
    MEMCPY(this->b, b_host.data(), sizeof(complex_t) * this->M, this->gpu, false);
    MEMCPY(this->W, W_host.data(), sizeof(complex_t) * sizeof_W, this->gpu, false);
    if(resize) {
        MEMCPY(this->spin_offset_list, spin_offset_list_host.data(), sizeof(unsigned int) * this->M, this->gpu, false);
        MEMCPY(this->W_offset_list, W_offset_list_host.data(), sizeof(unsigned int) * this->M, this->gpu, false);
        MEMCPY(this->param_offset_list, param_offset_list_host.data(), sizeof(unsigned int) * this->M, this->gpu, false);
        MEMCPY(this->string_length_list, string_length_list_host.data(), sizeof(unsigned int) * this->M, this->gpu, false);
        MEMCPY(this->hidden_spin_type_list, hidden_spin_type_host.data(), sizeof(unsigned int) * this->M, this->gpu, false);
    }
}

void PsiDynamical::get_active_params(complex<double>* result) const {
    for(auto i = 0u; i < this->N; i++) {
        result[i] = this->spin_weights[i];
    }

    auto param_offset = this->N + this->M;
    for(auto j = 0u; j < this->M; j++) {
        auto& link = this->links[j];

        result[this->N + j] = link.hidden_spin_weight;
        memcpy(&result[param_offset], link.weights.data(), sizeof(complex_t) * link.weights.size());
        param_offset += link.weights.size();
    }
}

void PsiDynamical::set_active_params(const complex<double>* new_params) {
    for(auto i = 0u; i < this->N; i++) {
        this->spin_weights[i] = new_params[i];
    }

    auto param_offset = this->N + this->M;
    for(auto j = 0u; j < this->M; j++) {
        auto& link = this->links[j];

        link.hidden_spin_weight = new_params[this->N + j];
        memcpy(link.weights.data(), &new_params[param_offset], sizeof(complex_t) * link.weights.size());
        param_offset += link.weights.size();
    }

    this->update(false);
}

void PsiDynamical::get_active_params_types(int* result) const {
    for(auto i = 0u; i < this->N; i++) {
        result[i] = -1;
    }

    auto param_offset = this->N + this->M;
    for(auto j = 0u; j < this->M; j++) {
        auto& link = this->links[j];

        result[this->N + j] = link.hidden_spin_type;
        for(auto i = 0u; i < link.weights.size(); i++) {
            result[param_offset + i] = link.hidden_spin_type;
        }
        param_offset += link.weights.size();
    }
}

} // namespace rbm_on_gpu
