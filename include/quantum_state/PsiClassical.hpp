#pragma once

#include "quantum_state/PsiDeepMin.hpp"

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

    HDINLINE
    complex_t log_psi_s(const Spins& spins) const {
        // complex_t tempHeff = complex_t(0.0, 0.0);

        // int omega1, omega2, omega3, omega4;
        // for (int j=0u; j<this->N; j++) {
        //     omega1 = (spins[j]*spins[(j+1)%this->N]+1)/2; // = 0,1
        //     tempHeff += this->W[omega1];

        //     omega2 =  2*(spins[j]*spins[(j+1)%this->N]+1)/2 +   (spins[(j+1)%this->N]*spins[(j+2)%this->N]+1)/2; // 0,1,2,3
        //     tempHeff += this->W[2+omega2];

        //     omega3 =  4*(spins[j]*spins[(j+1)%this->N]+1)/2 + 2*(spins[(j+1)%this->N]*spins[(j+2)%this->N]+1)/2 + 1*(spins[(j+2)%this->N]*spins[(j+3)%this->N]+1)/2; // 0-7
        //     tempHeff += this->W[6+omega3];

        //     omega4 =  8*(spins[j]*spins[(j+1)%this->N]+1)/2 + 4*(spins[(j+1)%this->N]*spins[(j+2)%this->N]+1)/2 + 2*(spins[(j+2)%this->N]*spins[(j+3)%this->N]+1)/2 + 1*(spins[(j+3)%this->N]*spins[(j+4)%this->N]+1)/2; // 0-15
        //     tempHeff += this->W[14+omega4];
        // }

        // // return tempHeff;
        // complex_t varW0 = this->W[this->num_params];
        // complex_t Heff_plaquetteComplex = complex_t(0.0, 0.0);

        // int omega;
        // for (auto j=0u; j<this->N; j++)
        //     {
        //     omega =   1*(spins[j]*spins[(j+1)%this->N]+1)/2 +        2*(spins[(j+1)%this->N]*spins[(j+2)%this->N]+1)/2 +  4*(spins[(j+2)%this->N]*spins[(j+3)%this->N]+1)/2 +   8*(spins[(j+3)%this->N]*spins[(j+4)%this->N]+1)/2
        //            + 16*(spins[(j+4)%this->N]*spins[(j+5)%this->N]+1)/2 + 32*(spins[(j+5)%this->N]*spins[(j+6)%this->N]+1)/2 + 64*(spins[(j+6)%this->N]*spins[(j+7)%this->N]+1)/2 ; // 0-127
        //     Heff_plaquetteComplex += complex_t(0.0, -1.0) * this->W[omega];

        //     // int omega =   1*(S[0][j][2]*S[0][(j+1)%L][2]+1)/2 +        2*(S[0][(j+1)%L][2]*S[0][(j+2)%L][2]+1)/2 +  4*(S[0][(j+2)%L][2]*S[0][(j+3)%L][2]+1)/2 +   8*(S[0][(j+3)%L][2]*S[0][(j+4)%L][2]+1)/2
        //     //         + 16*(S[0][(j+4)%L][2]*S[0][(j+5)%L][2]+1)/2 + 32*(S[0][(j+5)%L][2]*S[0][(j+6)%L][2]+1)/2 + 64*(S[0][(j+6)%L][2]*S[0][(j+7)%L][2]+1)/2 ; // 0-127
        //     // Heff_plaquetteComplex += (-I)*varW(omega);
        //     }

        // return varW0+Heff_plaquetteComplex;

        return this->log_psi_ptr[spins.configuration];
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

    PsiDeepMin* psi_neural;

    Array<complex_t> W_array;

    Array<double> alpha_array;
    Array<double> beta_array;

    bool free_quantum_axis;
    bool gpu;

public:
    // PsiClassical(const PsiClassical& other);

#ifdef __PYTHONCC__
    // inline PsiClassical(
    //     const string fname_base,
    //     const string fname_neural,
    //     const unsigned int N,
    //     const unsigned int num_params,
    //     const bool gpu
    // ) : psi_array(pow(2, N), false), W_array(num_params+1, gpu), alpha_array(1, false), beta_array(1, false), free_quantum_axis(false), gpu(gpu) {
    //     this->N = N;
    //     this->num_params = num_params;

    //     this->W_array.clear();
    //     this->psi_array.clear();

    //     this->init(fname_base, "Re");
    //     this->init(fname_base, "Im");

    //     this->W_array.update_device();
    //     this->W = this->W_array.data();

    //     this->prefactor = 1.0;

    //     psi_neural = new PsiDeepMin(fname_neural);
    //     this->psi_neural_kernel = this->psi_neural_kernel;
    // }

    inline PsiClassical(
        const xt::pytensor<std::complex<double>, 1u>& log_psi,
        const unsigned int N,
        const bool gpu
    ) : log_psi_array(log_psi, gpu), W_array(0, gpu), alpha_array(1, false), beta_array(1, false), free_quantum_axis(false), gpu(gpu) {
        this->N = N;

        this->log_psi_array.update_device();
        this->log_psi_ptr = this->log_psi_array.data();

        this->prefactor = 1.0;
        this->psi_neural = nullptr;
    }

    inline ~PsiClassical() {
        delete this->psi_neural;
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
