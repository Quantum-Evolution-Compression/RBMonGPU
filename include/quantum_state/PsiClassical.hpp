#pragma once

#include "quantum_state/PsiDeepMin.hpp"

#include "quantum_state/PsiClassicalHelper.hpp"


#include "Array.hpp"
#include "Spins.h"
#include "types.h"
#ifdef __CUDACC__
    #include "utils.kernel"
#endif
#include "cuda_complex.hpp"

#include <vector>
#include <complex>
#include <memory>
#include <cassert>

#ifdef __PYTHONCC__
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pytensor.hpp"

    using namespace std::complex_literals;
#endif // __PYTHONCC__


namespace rbm_on_gpu {

struct PsiClassicalAngles {
    PsiClassicalAngles() = default;

    template<typename Psi_t>
    HDINLINE void init(const Psi_t& psi, const PsiClassicalAngles& other) {
    }

    template<typename Psi_t>
    HDINLINE void init(const Psi_t& psi, const Spins& spins) {
    }

};

namespace kernel {

class PsiClassical {
public:
    unsigned int N;

    static constexpr unsigned int  max_N = MAX_SPINS;

    unsigned int   num_params;
    double          prefactor;

    complex_t* W;

    complex_t* log_psi_ptr;

// #ifdef __CUDACC__
    using Angles = rbm_on_gpu::PsiClassicalAngles;
    // using Derivatives = rbm_on_gpu::PsiClassicalDerivatives;

// #endif

public:

#ifndef __PYTHONCC__

    HDINLINE
    complex_t log_psi_s(const Spins& spins) const {
        #ifndef __CUDA_ARCH__
        std::vector<int> spins_vector(this->N);
        for(auto i = 0u; i < this->N; i++) {
            spins_vector[i] = spins[i];
        }

        const auto result = Peter::findHeffComplex(spins_vector);
        return complex_t(result.real(), result.imag());

        #else

        return complex_t(0.0, 0.0);

        #endif

        // return this->log_psi_ptr[spins.configuration];

    }



    // HDINLINE
    // complex_t log_psi_s_py(const vector<int>& spins) const {

    //     for(auto n = 0u; n < spins.size(); n++) {

    //     }
    // }

    HDINLINE
    void log_psi_s(complex_t& result, const Spins& spins, const Angles& angles) const {
        result = this->log_psi_s(spins);
    }

    HDINLINE
    void log_psi_s_real(double& result, const Spins& spins, const Angles& angles) const {
        result = this->log_psi_s(spins).real();
    }

    HDINLINE void flip_spin_of_jth_angle(
        const unsigned int j, const unsigned int position, const Spins& new_spins, Angles& angles
    ) const {
    }

    HDINLINE
    complex_t psi_s(const Spins& spins, const Angles& angles) const {
        #include "cuda_kernel_defines.h"

        SHARED complex_t log_psi;
        this->log_psi_s(log_psi, spins, angles);

        return exp(log_psi);
    }

#endif // __PYTHONCC__


    // complex<double> psi_s_std(const Spins& spins) const {
    //     return this->psi_s(spins).to_std();
    // }

    // HDINLINE
    // double probability_s(const Spins& spins, const Angles& angles) const {
    //     return exp(2.0 * (log(this->prefactor) + this->log_psi_s(spins, angles).real()));
    // }

    // double probability_s_py(const Spins& spins) const {
    //     return this->probability_s(spins);
    // }

    HDINLINE
    double probability_s(const double log_psi_s_real) const {
        return exp(2.0 * log_psi_s_real);
    }

    HDINLINE
    unsigned int get_num_spins() const {
        return this->N;
    }

    HDINLINE
    unsigned int get_num_angles() const {
        return 0u;
    }

    HDINLINE
    unsigned int get_width() const {
        return this->N;
    }

    HDINLINE
    static constexpr unsigned int get_max_spins() {
        return max_N;
    }

    HDINLINE
    unsigned int get_num_params() const {
        return this->num_params;
    }

    PsiClassical get_kernel() const {
        return *this;
    }

};

} // namespace kernel


class PsiClassical : public kernel::PsiClassical {
public:
    Array<complex_t> log_psi_array;
    Array<complex_t> W_array;

    Array<double> alpha_array;
    Array<double> beta_array;

    bool free_quantum_axis;
    bool gpu;

public:
    // PsiClassical(const PsiClassical& other);

#ifdef __PYTHONCC__
    inline PsiClassical(
        const string directory,
        const int index,
        const unsigned int N,
        const bool gpu
    ) : log_psi_array(1, false), W_array(1, false), alpha_array(1, false), beta_array(1, false), free_quantum_axis(false), gpu(gpu) {
        this->N = N;

        Peter::loadVP(directory, index, "Re");
        Peter::loadVP(directory, index, "Im");
        Peter::Compress_Load(directory, index);
        Peter::load_neural_network(directory, index);

        this->prefactor = 1.0;
    }

    // inline PsiClassical(
    //     const xt::pytensor<std::complex<double>, 1u>& log_psi,
    //     const unsigned int N,
    //     const bool gpu
    // ) : log_psi_array(log_psi, gpu), W_array(0, gpu), alpha_array(1, false), beta_array(1, false), free_quantum_axis(false), gpu(gpu) {
    //     this->N = N;

    //     this->log_psi_array.update_device();
    //     this->log_psi_ptr = this->log_psi_array.data();

    //     this->prefactor = 1.0;
    // }

    inline ~PsiClassical() {
    }

    void init(const string& fname_base, const std::string& ReIm) {
        string filenamePos = fname_base + ReIm+".csv";
        std::ifstream filePos;
        filePos.open (filenamePos.c_str());

        std::string temp;
        complex_t I(0.0, 1.0);

        for (auto i=0u; i<this->num_params; i++) {
            getline (filePos, temp);
            if (ReIm.find("Re") != std::string::npos) this->W_array[i] +=   atof(temp.c_str());
            if (ReIm.find("Im") != std::string::npos) this->W_array[i] += I*atof(temp.c_str());
        }

        getline (filePos, temp);
        if (ReIm.find("Re") != std::string::npos) this->W_array[this->num_params] +=   atof(temp.c_str()); // "dumb" variational parameter for normalization
        if (ReIm.find("Im") != std::string::npos) this->W_array[this->num_params] += I*atof(temp.c_str());

        filePos.close();
    }

    // xt::pytensor<complex<double>, 1> as_vector_py() const {
    //     auto result = xt::pytensor<complex<double>, 1>(
    //         std::array<long int, 1>({static_cast<long int>(pow(2, this->N))})
    //     );
    //     this->as_vector(result.data());

    //     return result;
    // }

    // PsiClassical copy() const {
    //     return *this;
    // }

    // xt::pytensor<complex<double>, 1> get_params_py() const {
    //     auto result = xt::pytensor<complex<double>, 1>(
    //         std::array<long int, 1>({static_cast<long int>(this->num_params)})
    //     );
    //     this->get_params(result.data());

    //     return result;
    // }

    // void set_params_py(const xt::pytensor<complex<double>, 1>& new_params) {
    //     this->set_params(new_params.data());
    // }

    unsigned int get_num_params_py() const {
        return this->get_num_params();
    }

#endif // __PYTHONCC__

    // void as_vector(complex<double>* result) const;
    // void O_k_vector(complex<double>* result, const Spins& spins) const;
    // double norm_function(const ExactSummation& exact_summation) const;
    // complex<double> log_psi_s_std(const Spins& spins);

    // void get_params(complex<double>* result) const;
    // void set_params(const complex<double>* new_params);

    // void update_kernel();
};

} // namespace rbm_on_gpu
