#include "network_functions/ExpectationValue.hpp"
#include "spin_ensembles.hpp"
#include "quantum_states.hpp"
#include "Array.hpp"


namespace rbm_on_gpu {

ExpectationValue::ExpectationValue(const bool gpu) : gpu(gpu) {
    MALLOC(this->result, sizeof(complex_t), this->gpu);
}

ExpectationValue::~ExpectationValue() noexcept(false) {
    FREE(this->result, this->gpu);
}

template<typename Psi_t, typename SpinEnsemble>
complex<double> ExpectationValue::operator()(
    const Psi_t& psi, const Operator& operator_, const SpinEnsemble& spin_ensemble
) const {

    complex<double> result_host;
    complex_t* this_result = this->result;
    const auto psi_kernel = psi.get_kernel();
    const auto operator_kernel = operator_.get_kernel();

    if(this->gpu) {
        CUDA_CHECK(cudaMemset(this->result, 0, sizeof(complex_t)))

        spin_ensemble.foreach(
            psi,
            [=] __device__ (
                const unsigned int spin_index,
                const Spins spins,
                const complex_t log_psi,
                const typename Psi_t::Angles& angles,
                const double weight
            ) {
                __shared__ complex_t local_energy;
                operator_kernel.local_energy(local_energy, psi_kernel, spins, log_psi, angles);
                if(threadIdx.x == 0) {
                    atomicAdd(this_result, weight * local_energy);
                }
            }
        );

        CUDA_CHECK(cudaMemcpy(&result_host, this->result, sizeof(complex_t), cudaMemcpyDeviceToHost))
    }
    else {
        *this->result = complex_t(0.0, 0.0);

        spin_ensemble.foreach(
            psi,
            [=] __device__ __host__ (
                const unsigned int spin_index,
                const Spins spins,
                const complex_t log_psi,
                const typename Psi_t::Angles& angles,
                const double weight
            ) {
                complex_t local_energy;
                operator_kernel.local_energy(local_energy, psi_kernel, spins, log_psi, angles);
                *this_result += weight * local_energy;
            }
        );

        result_host = this->result->to_std();
    }

    result_host *= 1.0 / spin_ensemble.get_num_steps();

    return result_host;
}

template<typename Psi_t, typename SpinEnsemble>
pair<double, complex<double>> ExpectationValue::fluctuation(const Psi_t& psi, const Operator& operator_, const SpinEnsemble& spin_ensemble) const {
    const auto psi_kernel = psi.get_kernel();
    const auto op_kernel = operator_.get_kernel();

    Array<complex_t> E_loc_avg(1, psi.gpu);
    Array<double> E_loc2_avg(1, psi.gpu);

    E_loc_avg.clear();
    E_loc2_avg.clear();

    auto E_loc_ptr = E_loc_avg.data();
    auto E_loc2_ptr = E_loc2_avg.data();

    spin_ensemble.foreach(
        psi,
        [=] __device__ __host__ (
            const unsigned int spin_index,
            const Spins spins,
            const complex_t log_psi,
            const typename Psi_t::Angles& angles,
            const double weight
        ) {
            #include "cuda_kernel_defines.h"

            SHARED complex_t local_energy;
            op_kernel.local_energy(local_energy, psi_kernel, spins, log_psi, angles);

            #ifdef __CUDA_ARCH__
            if(threadIdx.x == 0)
            #endif
            {
                generic_atomicAdd(E_loc_ptr, weight * local_energy);
                generic_atomicAdd(E_loc2_ptr, weight * norm(local_energy));
            }
        }
    );

    E_loc_avg.update_host();
    E_loc2_avg.update_host();

    const auto E_loc = E_loc_avg.front() * (1.0 / spin_ensemble.get_num_steps());
    const auto E_loc2 = E_loc2_avg.front() * (1.0 / spin_ensemble.get_num_steps());

    return {sqrt(E_loc2 - norm(E_loc)), E_loc.to_std()};
}

template<typename Psi_t, typename SpinEnsemble>
complex<double> ExpectationValue::gradient(
    complex<double>* result, const Psi_t& psi, const Operator& operator_, const SpinEnsemble& spin_ensemble
) const {
    const auto O_k_length = psi.get_num_params();
    const auto psi_kernel = psi.get_kernel();
    const auto op_kernel = operator_.get_kernel();

    Array<complex_t> E_loc_avg(1, psi.gpu);
    Array<complex_t> O_k_avg(O_k_length, psi.gpu);
    Array<complex_t> E_loc_O_k_avg(O_k_length, psi.gpu);
    Array<complex_t> E_loc_k_avg(O_k_length, psi.gpu);

    E_loc_avg.clear();
    O_k_avg.clear();
    E_loc_O_k_avg.clear();
    E_loc_k_avg.clear();

    auto E_loc_ptr = E_loc_avg.data();
    auto O_k_ptr = O_k_avg.data();
    auto E_loc_O_k_ptr = E_loc_O_k_avg.data();
    auto E_loc_k_ptr = E_loc_k_avg.data();

    spin_ensemble.foreach(
        psi,
        [=] __device__ __host__ (
            const unsigned int spin_index,
            const Spins spins,
            const complex_t log_psi,
            typename Psi_t::Angles& angles,
            const double weight
        ) {
            #include "cuda_kernel_defines.h"

            SHARED complex_t local_energy;
            op_kernel.local_energy(local_energy, psi_kernel, spins, log_psi, angles);

            SINGLE
            {
                generic_atomicAdd(E_loc_ptr, weight * local_energy);
            }

            psi_kernel.foreach_O_k(
                spins,
                angles,
                [&](const unsigned int k, const complex_t& O_k_element) {
                    generic_atomicAdd(&O_k_ptr[k], weight * O_k_element);
                    generic_atomicAdd(&E_loc_O_k_ptr[k], weight * local_energy * conj(O_k_element));
                }
            );

            SYNC;

            op_kernel.foreach_E_k_s_prime(
                psi_kernel, spins, log_psi, angles, [&](const unsigned int k, const complex_t& E_k_s_prime) {
                    generic_atomicAdd(&E_loc_k_ptr[k], weight * E_k_s_prime);
                }
            );
        }
    );

    E_loc_avg.update_host();
    O_k_avg.update_host();
    E_loc_O_k_avg.update_host();
    E_loc_k_avg.update_host();

    E_loc_avg.front() *= 1.0 / spin_ensemble.get_num_steps();

    for(auto k = 0u; k < O_k_length; k++) {
        O_k_avg[k] *= 1.0 / spin_ensemble.get_num_steps();
        E_loc_O_k_avg[k] *= 1.0 / spin_ensemble.get_num_steps();
        E_loc_k_avg[k] *= 1.0 / spin_ensemble.get_num_steps();

        result[k] = (
            E_loc_O_k_avg[k] + conj(E_loc_k_avg[k]) - 2.0 * E_loc_avg.front() * conj(O_k_avg[k])
        ).to_std();
    }

    return E_loc_avg.front().to_std();
}

template<typename Psi_t, typename SpinEnsemble>
void ExpectationValue::fluctuation_gradient(complex<double>* result, const Psi_t& psi, const Operator& operator_, const SpinEnsemble& spin_ensemble) const {
    const auto O_k_length = psi.get_num_params();
    const auto psi_kernel = psi.get_kernel();
    const auto op_kernel = operator_.get_kernel();

    Array<complex_t> E_loc_avg(1, psi.gpu);
    Array<double> E_loc2_avg(1, psi.gpu);
    Array<complex_t> E_loc_E_loc_k_avg(O_k_length, psi.gpu);
    Array<complex_t> O_k_avg(O_k_length, psi.gpu);
    Array<complex_t> E_loc_O_k_avg(O_k_length, psi.gpu);
    Array<complex_t> E_loc_k_avg(O_k_length, psi.gpu);

    E_loc_avg.clear();
    E_loc2_avg.clear();
    E_loc_E_loc_k_avg.clear();
    O_k_avg.clear();
    E_loc_O_k_avg.clear();
    E_loc_k_avg.clear();

    auto E_loc_ptr = E_loc_avg.data();
    auto E_loc2_ptr = E_loc2_avg.data();
    auto E_loc_E_loc_k_ptr = E_loc_E_loc_k_avg.data();
    auto O_k_ptr = O_k_avg.data();
    auto E_loc_O_k_ptr = E_loc_O_k_avg.data();
    auto E_loc_k_ptr = E_loc_k_avg.data();

    spin_ensemble.foreach(
        psi,
        [=] __device__ __host__ (
            const unsigned int spin_index,
            const Spins spins,
            const complex_t log_psi,
            typename Psi_t::Angles& angles,
            const double weight
        ) {
            #include "cuda_kernel_defines.h"

            SHARED complex_t local_energy;
            op_kernel.local_energy(local_energy, psi_kernel, spins, log_psi, angles);

            SINGLE
            {
                generic_atomicAdd(E_loc_ptr, weight * local_energy);
                generic_atomicAdd(E_loc2_ptr, weight * norm(local_energy));
            }

            psi_kernel.foreach_O_k(
                spins,
                angles,
                [&](const unsigned int k, const complex_t& O_k_element) {
                    generic_atomicAdd(&O_k_ptr[k], weight * O_k_element);
                    generic_atomicAdd(&E_loc_O_k_ptr[k], weight * local_energy * conj(O_k_element));
                }
            );

            SYNC;

            op_kernel.foreach_E_k_s_prime(
                psi_kernel, spins, log_psi, angles, [&](const unsigned int k, const complex_t& E_k_s_prime) {
                    generic_atomicAdd(&E_loc_k_ptr[k], weight * E_k_s_prime);
                    generic_atomicAdd(&E_loc_E_loc_k_ptr[k], weight * (conj(local_energy) * E_k_s_prime));
                }
            );
        }
    );

    E_loc_avg.update_host();
    E_loc2_avg.update_host();
    E_loc_E_loc_k_avg.update_host();
    O_k_avg.update_host();
    E_loc_O_k_avg.update_host();
    E_loc_k_avg.update_host();

    const auto E_loc = E_loc_avg.front() * (1.0 / spin_ensemble.get_num_steps());
    const auto E_loc2 = E_loc2_avg.front() * (1.0 / spin_ensemble.get_num_steps());

    const auto fluctuation = sqrt(E_loc2 - norm(E_loc));

    for(auto k = 0u; k < O_k_length; k++) {
        const auto E_loc_E_loc_k = E_loc_E_loc_k_avg[k] * (1.0 / spin_ensemble.get_num_steps());
        const auto O_k = O_k_avg[k] * (1.0 / spin_ensemble.get_num_steps());
        const auto E_loc_O_k = E_loc_O_k_avg[k] * (1.0 / spin_ensemble.get_num_steps());
        const auto E_loc_k = E_loc_k_avg[k] * (1.0 / spin_ensemble.get_num_steps());

        result[k] = 1.0 / (2.0 * fluctuation) * (
            2.0 * conj(E_loc_E_loc_k - conj(E_loc) * E_loc_k) - 2.0 * (conj(E_loc) * E_loc_O_k)
            + 2.0 * conj(O_k) * (2.0 * norm(E_loc) - E_loc2)
        ).to_std();
    }
}

template<typename Psi_t, typename SpinEnsemble>
vector<complex<double>> ExpectationValue::difference(
    const Psi_t& psi, const Psi_t& psi_prime, const vector<Operator>& operator_list_host, const SpinEnsemble& spin_ensemble
) const {
    const auto length = operator_list_host.size();


    Operator* operator_list;
    complex_t* a_list;
    complex_t* b_list;
    double* probability_ratio_avg;

    MALLOC(operator_list, sizeof(Operator) * length, psi.gpu);
    MEMCPY(operator_list, operator_list_host.data(), sizeof(Operator) * length, psi.gpu, false);

    MALLOC(a_list, sizeof(complex_t) * length, psi.gpu);
    MALLOC(b_list, sizeof(complex_t) * length, psi.gpu);
    MEMSET(a_list, 0, sizeof(complex_t) * length, psi.gpu);
    MEMSET(b_list, 0, sizeof(complex_t) * length, psi.gpu);
    MALLOC(probability_ratio_avg, sizeof(double), psi.gpu);
    MEMSET(probability_ratio_avg, 0, sizeof(double), psi.gpu);

    const auto psi_kernel = psi.get_kernel();
    const auto psi_prime_kernel = psi_prime.get_kernel();

    spin_ensemble.foreach(
        psi,
        [=] __device__ __host__ (
            const unsigned int spin_index,
            const Spins spins,
            const complex_t log_psi,
            const typename Psi_t::Angles& angles,
            const double weight
        ) {
            #include "cuda_kernel_defines.h"

            SHARED typename Psi_t::Angles angles_prime;
            angles_prime.init(psi_prime_kernel, spins);

            SHARED complex_t log_psi_prime;
            psi_prime_kernel.log_psi_s(log_psi_prime, spins, angles_prime);

            double probability_ratio;
            SINGLE
            {
                probability_ratio = exp(2.0 * (log_psi.real() - log_psi_prime.real()));
            }

            SHARED complex_t local_energy;
            SHARED complex_t local_energy_prime;

            for(auto i = 0u; i < length; i++) {
                operator_list[i].local_energy(local_energy, psi_kernel, spins, log_psi, angles);
                operator_list[i].local_energy(local_energy_prime, psi_prime_kernel, spins, log_psi_prime, angles_prime);

                SINGLE
                {
                    generic_atomicAdd(&a_list[i], weight * local_energy_prime);
                    generic_atomicAdd(&b_list[i], weight * probability_ratio * local_energy);
                }
            }

            SINGLE
            {
                generic_atomicAdd(probability_ratio_avg, weight * probability_ratio);
            }
        }
    );

    vector<complex<double>> a_list_host(length);
    vector<complex<double>> b_list_host(length);
    double probability_ratio_host;

    MEMCPY_TO_HOST(a_list_host.data(), a_list, sizeof(complex_t) * length, psi.gpu);
    MEMCPY_TO_HOST(b_list_host.data(), b_list, sizeof(complex_t) * length, psi.gpu);
    MEMCPY_TO_HOST(&probability_ratio_host, probability_ratio_avg, sizeof(double), psi.gpu);

    FREE(operator_list, psi.gpu);
    FREE(a_list, psi.gpu);
    FREE(b_list, psi.gpu);
    FREE(probability_ratio_avg, psi.gpu);

    for(auto& a : a_list_host) {
        a *= 1.0 / spin_ensemble.get_num_steps();
    }
    for(auto& b : b_list_host) {
        b *= 1.0 / spin_ensemble.get_num_steps();
    }
    probability_ratio_host *= 1.0 / spin_ensemble.get_num_steps();

    vector<complex<double>> result(length);
    for(auto i = 0u; i < length; i++) {
        result[i] = a_list_host[i] - b_list_host[i] / probability_ratio_host;
    }

    return result;
}

template<typename Psi_t, typename SpinEnsemble>
vector<complex<double>> ExpectationValue::operator() (
    const Psi_t& psi,
    const vector<Operator>& operator_list_host,
    const SpinEnsemble& spin_ensemble
) const {
    const auto length = operator_list_host.size();
    vector<complex<double>> result_host(length);

    const Operator* operator_list;
    complex_t* result;

    if(this->gpu) {
        CUDA_CHECK(cudaMalloc(&operator_list, sizeof(Operator) * length));
        CUDA_CHECK(cudaMemcpy((void*)operator_list, operator_list_host.data(), sizeof(Operator) * length, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&result, sizeof(complex_t) * length));
        CUDA_CHECK(cudaMemset(result, 0, sizeof(complex_t) * length));
    }
    else {
        operator_list = operator_list_host.data();
        result_host.assign(length, complex<double>(0.0, 0.0));
        result = (complex_t*)result_host.data();
    }

    const auto psi_kernel = psi.get_kernel();

    spin_ensemble.foreach(
        psi,
        [=] __host__ __device__ (
            const unsigned int spin_index,
            const Spins spins,
            const complex_t log_psi,
            const typename Psi_t::Angles& angles,
            const double weight
        ) {
            #ifdef __CUDA_ARCH__

            __shared__ complex_t local_energy;
            for(auto i = 0u; i < length; i++) {
                operator_list[i].local_energy(local_energy, psi_kernel, spins, log_psi, angles);
                if(threadIdx.x == 0) {
                    atomicAdd(&result[i], weight * local_energy);
                }
            }

            #else

            complex_t local_energy;
            for(auto i = 0u; i < length; i++) {
                operator_list[i].local_energy(local_energy, psi_kernel, spins, log_psi, angles);
                result[i] += weight * local_energy;
            }

            #endif
        }
    );

    if(this->gpu) {
        CUDA_CHECK(cudaMemcpy(result_host.data(), result, sizeof(complex_t) * length, cudaMemcpyDeviceToHost))

        CUDA_CHECK(cudaFree((void*)operator_list));
        CUDA_CHECK(cudaFree(result));
    }

    for(auto& r : result_host) {
        r *= 1.0 / spin_ensemble.get_num_steps();
    }

    return result_host;
}


#ifdef ENABLE_MONTE_CARLO

#ifdef ENABLE_PSI

template complex<double> ExpectationValue::operator()(const Psi& psi, const Operator& operator_, const MonteCarloLoop&) const;
template pair<double, complex<double>> ExpectationValue::fluctuation(const Psi&, const Operator&, const MonteCarloLoop&) const;
template complex<double> ExpectationValue::gradient(complex<double>*, const Psi&, const Operator&, const MonteCarloLoop&) const;
template void ExpectationValue::fluctuation_gradient(complex<double>*, const Psi&, const Operator&, const MonteCarloLoop&) const;
template vector<complex<double>> ExpectationValue::difference(const Psi&, const Psi&, const vector<Operator>&, const MonteCarloLoop&) const;
template vector<complex<double>> ExpectationValue::operator()(
    const Psi& psi, const vector<Operator>& operator_, const MonteCarloLoop&
) const;

#endif

#ifdef ENABLE_PSI_DEEP

template complex<double> ExpectationValue::operator()(const PsiDeep& psi, const Operator& operator_, const MonteCarloLoop&) const;
template pair<double, complex<double>> ExpectationValue::fluctuation(const PsiDeep&, const Operator&, const MonteCarloLoop&) const;
template complex<double> ExpectationValue::gradient(complex<double>*, const PsiDeep&, const Operator&, const MonteCarloLoop&) const;
template void ExpectationValue::fluctuation_gradient(complex<double>*, const PsiDeep&, const Operator&, const MonteCarloLoop&) const;
template vector<complex<double>> ExpectationValue::operator()(
    const PsiDeep& psi, const vector<Operator>& operator_, const MonteCarloLoop&
) const;

#endif

#ifdef ENABLE_PSI_PAIR

template complex<double> ExpectationValue::operator()(const PsiPair& psi, const Operator& operator_, const MonteCarloLoop&) const;
// template pair<double, complex<double>> ExpectationValue::fluctuation(const PsiPair&, const Operator&, const MonteCarloLoop&) const;
// template complex<double> ExpectationValue::gradient(complex<double>*, const PsiPair&, const Operator&, const MonteCarloLoop&) const;
template vector<complex<double>> ExpectationValue::operator()(
    const PsiPair& psi, const vector<Operator>& operator_, const MonteCarloLoop&
) const;

#endif

#ifdef ENABLE_PSI_HAMILTONIAN
template complex<double> ExpectationValue::operator()(const PsiHamiltonian& psi, const Operator& operator_, const MonteCarloLoop&) const;
#endif

#endif


#ifdef ENABLE_EXACT_SUMMATION

#ifdef ENABLE_PSI

template complex<double> ExpectationValue::operator()(const Psi& psi, const Operator& operator_, const ExactSummation&) const;
template pair<double, complex<double>> ExpectationValue::fluctuation(const Psi&, const Operator&, const ExactSummation&) const;
template complex<double> ExpectationValue::gradient(complex<double>*, const Psi&, const Operator&, const ExactSummation&) const;
template void ExpectationValue::fluctuation_gradient(complex<double>*, const Psi&, const Operator&, const ExactSummation&) const;
template vector<complex<double>> ExpectationValue::difference(const Psi&, const Psi&, const vector<Operator>&, const ExactSummation&) const;
template vector<complex<double>> ExpectationValue::operator()(
    const Psi& psi, const vector<Operator>& operator_, const ExactSummation&
) const;

#endif

#ifdef ENABLE_PSI_DEEP

template complex<double> ExpectationValue::operator()(const PsiDeep& psi, const Operator& operator_, const ExactSummation&) const;
template pair<double, complex<double>> ExpectationValue::fluctuation(const PsiDeep&, const Operator&, const ExactSummation&) const;
template complex<double> ExpectationValue::gradient(complex<double>*, const PsiDeep&, const Operator&, const ExactSummation&) const;
template void ExpectationValue::fluctuation_gradient(complex<double>*, const PsiDeep&, const Operator&, const ExactSummation&) const;
template vector<complex<double>> ExpectationValue::operator()(
    const PsiDeep& psi, const vector<Operator>& operator_, const ExactSummation&
) const;

#endif

#ifdef ENABLE_PSI_PAIR

template complex<double> ExpectationValue::operator()(const PsiPair& psi, const Operator& operator_, const ExactSummation&) const;
// template pair<double, complex<double>> ExpectationValue::fluctuation(const PsiPair&, const Operator&, const ExactSummation&) const;
// template complex<double> ExpectationValue::gradient(complex<double>*, const PsiPair&, const Operator&, const ExactSummation&) const;
// template void ExpectationValue::fluctuation_gradient(complex<double>*, const PsiPair&, const Operator&, const ExactSummation&) const;
template vector<complex<double>> ExpectationValue::operator()(
    const PsiPair& psi, const vector<Operator>& operator_, const ExactSummation&
) const;

#endif

#ifdef ENABLE_PSI_HAMILTONIAN
template complex<double> ExpectationValue::operator()(const PsiHamiltonian& psi, const Operator& operator_, const ExactSummation&) const;
#endif

#endif


} // namespace rbm_on_gpu
