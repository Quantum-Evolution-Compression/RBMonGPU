#pragma once

#include "quantum_state/Psi.hpp"
#include "quantum_state/psi_functions.hpp"
#include "quantum_state/PsiDynamicalCache.hpp"
#include "Spins.h"
#include "types.h"
#include "cuda_complex.hpp"

#include <vector>
#include <complex>
#include <memory>
#include <cassert>

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pytensor.hpp"

    using namespace std::complex_literals;
#endif // __CUDACC__


namespace rbm_on_gpu {

namespace kernel {

class PsiDynamical : public Psi {
public:
    unsigned int num_params;

    unsigned int* spin_offset_list;
    unsigned int* W_offset_list;
    unsigned int* param_offset_list;
    unsigned int* string_length_list;
    int* hidden_spin_type_list;


#ifdef __CUDACC__
    using Cache = rbm_on_gpu::PsiDynamicalCache;
#endif

public:

#ifdef __CUDACC__

    HDINLINE
    complex_t angle(const unsigned int j, const Spins& spins) const {
        complex_t result = this->b[j];

        const auto W_j = &(this->W[W_offset_list[j]]);
        const auto string_length = this->string_length_list[j];
        const auto spin_offset = this->spin_offset_list[j];
        for(auto i = 0u; i < string_length; i++) {
            result += W_j[i] * spins[(spin_offset + i) % this->N];
        }

        return result;
    }

    HDINLINE
    void init_angles(complex_t* angles, const Spins& spins) const {
        #ifdef __CUDA_ARCH__
            const auto j = threadIdx.x;
            if(j < this->M)
        #else
            for(auto j = 0u; j < this->M; j++)
        #endif
        {
            angles[j] = this->angle(j, spins);
        }
    }

    HDINLINE
    complex_t log_psi_s(const Spins& spins) const {
        complex_t result(0.0, 0.0);
        for(unsigned int i = 0; i < this->N; i++) {
            result += this->a[i] * spins[i];
        }
        for(unsigned int j = 0; j < this->M; j++) {
            auto angle = this->angle(j, spins);
            if(this->hidden_spin_type_list[j] == 1) {
                angle = angle * angle;
            }
            result += /*this->n[j] **/ my_logcosh(angle);
        }

        return result;
    }

    HDINLINE
    void log_psi_s(complex_t& result, const Spins& spins, const complex_t* angle_ptr) const {
        // CAUTION: 'result' has to be a shared variable.
        // j = threadIdx.x

        #ifdef __CUDA_ARCH__

        auto summand = complex_t(0.0, 0.0);
        if(threadIdx.x < this->N) {
            summand += this->a[threadIdx.x] * spins[threadIdx.x];
        }
        if(threadIdx.x < this->M) {
            auto angle = angle_ptr[threadIdx.x];
            if(this->hidden_spin_type_list[threadIdx.x] == 1) {
                angle = angle * angle;
            }
            summand += my_logcosh(angle);
        }

        tree_sum(result, this->M, summand);

        #else

        result = complex_t(0.0, 0.0);
        for(auto i = 0u; i < this->N; i++) {
            result += this->a[i] * spins[i];
        }
        for(auto j = 0u; j < this->M; j++) {
            auto angle = angle_ptr[j];
            if(this->hidden_spin_type_list[j] == 1) {
                angle = angle * angle;
            }
            result += my_logcosh(angle);
        }

        #endif
    }

    HDINLINE
    void log_psi_s_real(double& result, const Spins& spins, const complex_t* angle_ptr) const {
        // CAUTION: 'result' has to be a shared variable.
        // j = threadIdx.x

        #ifdef __CUDA_ARCH__

        auto summand = 0.0;
        if(threadIdx.x < this->N) {
            summand += this->a[threadIdx.x].real() * spins[threadIdx.x];
        }
        if(threadIdx.x < this->M) {
            auto angle = angle_ptr[threadIdx.x];
            if(this->hidden_spin_type_list[threadIdx.x] == 1) {
                angle = angle * angle;
            }
            summand += my_logcosh(angle).real();
        }

        tree_sum(result, this->M, summand);

        #else

        result = 0.0;
        for(auto i = 0u; i < this->N; i++) {
            result += this->a[i].real() * spins[i];
        }
        for(auto j = 0u; j < this->M; j++) {
            auto angle = angle_ptr[j];
            if(this->hidden_spin_type_list[j] == 1) {
                angle = angle * angle;
            }
            result += my_logcosh(angle).real();
        }

        #endif
    }

    HDINLINE complex_t flip_spin_of_jth_angle(
        const complex_t* angles, const unsigned int j, const unsigned int position, const Spins& new_spins
    ) const {
        // return this->angle(j, new_spins);
        complex_t new_angle;
        if(j < this->get_num_angles()) {
            new_angle = angles[j];
            const auto relative_position = (position - this->spin_offset_list[j] + this->N) % this->N;
            if(relative_position < this->string_length_list[j]) {
                new_angle += 2.0 * new_spins[position] * this->W[this->W_offset_list[j] + relative_position];
            }
        }

        return new_angle;
    }

    template<typename Function>
    HDINLINE
    void foreach_O_k(const Spins& spins, const Cache& cache, Function function) const {
        #ifdef __CUDA_ARCH__
        const auto j = threadIdx.x;
        if(j < this->M)
        #else
        for(auto j = 0u; j < this->M; j++)
        #endif
        {
            if(j < this->N) {
                function(j, complex_t(spins[j], 0.0));
            }

            const auto tanh_angle = cache.tanh_angles[j];
            const auto hidden_spin_type = this->hidden_spin_type_list[j];
            const auto factor = hidden_spin_type == 1 ? 2.0 * cache.angles[j] : complex_t(1.0, 0.0);

            function(this->N + j, tanh_angle * factor);

            const auto spin_offset = this->spin_offset_list[j];
            const auto param_offset = this->param_offset_list[j];
            for(auto i = 0u; i < this->string_length_list[j]; i++) {
                function(param_offset + i, tanh_angle * factor * spins[(spin_offset + i) % this->N]);
            }
        }
    }

#endif // __CUDACC__

    HDINLINE
    unsigned int get_num_params() const {
        return this->num_params;
    }

    HDINLINE
    unsigned int get_num_active_params() const {
        return this->num_params;
    }

    inline const PsiDynamical& get_kernel() const {
        return *this;
    }
};

} // namespace kernel


class PsiDynamical : public kernel::PsiDynamical {
public:
    using clist = vector<complex<double>>;

    struct Link {
        unsigned int    first_spin;
        clist           weights;
        complex<double> hidden_spin_weight;
        int             hidden_spin_type;
    };

    clist spin_weights;
    vector<Link> links;
    bool gpu;

public:
    PsiDynamical(const clist& spin_weights, const bool gpu=true);

    PsiDynamical(const PsiDynamical& other);
    ~PsiDynamical() noexcept(false);

    void add_hidden_spin(
        const unsigned int first_spin, const clist& link_weights, const complex<double>& hidden_spin_weight, const int hidden_spin_type
    );
    void update(bool resize);

    inline bool on_gpu() const {
        return this->gpu;
    }

#ifdef __PYTHONCC__
    xt::pytensor<complex<double>, 1> as_vector_py() const {
        auto result = xt::pytensor<complex<double>, 1>(
            std::array<long int, 1>({static_cast<long int>(pow(2, this->N))})
        );
        this->as_vector(result.raw_data());

        return result;
    }

    xt::pytensor<complex<double>, 1> O_k_vector_py(const Spins& spins) const {
        auto result = xt::pytensor<complex<double>, 1>(
            std::array<long int, 1>({static_cast<long int>(this->num_active_params)})
        );
        this->O_k_vector(result.raw_data(), spins);

        return result;
    }

    xt::pytensor<complex<double>, 1> a_py() const {
        auto result = xt::pytensor<complex<double>, 1>(
            std::array<long int, 1>({static_cast<long int>(this->N)})
        );

        memcpy(result.raw_data(), this->spin_weights.data(), sizeof(complex_t) * this->N);

        return result;
    }

    void set_a_py(const xt::pytensor<complex<double>, 1>& new_a) {
        memcpy(this->spin_weights.data(), new_a.raw_data(), sizeof(complex_t) * this->N);
    }

    xt::pytensor<complex<double>, 1> b_py() const {
        auto result = xt::pytensor<complex<double>, 1>(
            std::array<long int, 1>({static_cast<long int>(this->M)})
        );

        auto j = 0u;
        for(const auto& link : this->links) {
            result.raw_data()[j] = link.hidden_spin_weight;

            j++;
        }

        return result;
    }

    xt::pytensor<complex<double>, 2> dense_W_py() const {
        auto result = xt::pytensor<complex<double>, 2>(
            std::array<long int, 2>(
                {static_cast<long int>(this->N), static_cast<long int>(this->M)}
            )
        );
        this->dense_W(result.raw_data());

        return result;
    }

    xt::pytensor<complex<double>, 1> get_active_params_py() const {
        auto result = xt::pytensor<complex<double>, 1>(
            std::array<long int, 1>({static_cast<long int>(this->num_active_params)})
        );
        this->get_active_params(result.raw_data());

        return result;
    }

    void set_active_params_py(const xt::pytensor<complex<double>, 1>& new_params) {
        this->set_active_params(new_params.raw_data());
    }

    xt::pytensor<int, 1> get_active_params_types_py() const {
        auto result = xt::pytensor<int, 1>(
            std::array<long int, 1>({static_cast<long int>(this->num_active_params)})
        );
        this->get_active_params_types(result.raw_data());

        return result;
    }

    PsiDynamical copy() const {
        return *this;
    }

#endif // __CUDACC__

    void as_vector(complex<double>* result) const;
    void O_k_vector(complex<double>* result, const Spins& spins) const;
    double norm_function() const;
    void dense_W(complex<double>* result) const;

    unsigned int get_num_params_py() const {
        return this->get_num_params();
    }

    void get_active_params(complex<double>* result) const;
    void set_active_params(const complex<double>* new_params);

    void get_active_params_types(int* result) const;

private:

    void init(const complex<double>* a, const unsigned int N, const bool a_on_gpu, const double prefactor=1.0);
    void clear_hidden_spins();
    unsigned int sizeof_W() const;
};

} // namespace rbm_on_gpu
