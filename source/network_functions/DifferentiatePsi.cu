#include "network_functions/DifferentiatePsi.hpp"
#include "spin_ensembles/ExactSummation.hpp"
#include "spin_ensembles/MonteCarloLoop.hpp"
#include "zMinResQLP.h"

#include <cstring>


namespace rbm_on_gpu {


__global__ void kernel_divide_array_by_n(complex_t* array, size_t array_size, double one_over_n) {
    const auto linear_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(linear_idx < array_size) {
        array[linear_idx] *= one_over_n;
    }
}


DifferentiatePsi::DifferentiatePsi(const unsigned int O_k_length, const bool gpu)
  : gpu(gpu) {
    this->O_k_length = O_k_length;
    this->spin_history_ptr = nullptr;
    this->psi_history_ptr = nullptr;

    MALLOC(this->forces, sizeof(complex_t) * this->O_k_length, this->gpu);
    MALLOC(this->O_k_avg, sizeof(complex_t) * this->O_k_length, this->gpu);
    MALLOC(this->E_local_avg, sizeof(complex_t), this->gpu);
    MALLOC(this->E_local_O_k_avg, sizeof(complex_t) * this->O_k_length, this->gpu);

    if(gpu) {
        MALLOC(this->x, sizeof(complex_t) * this->O_k_length, this->gpu);
        MALLOC(this->Ax, sizeof(complex_t) * this->O_k_length, this->gpu);

        this->forces_host = new complex<double>[this->O_k_length];
    }
    MALLOC(this->O_k_avg_x, sizeof(complex_t), this->gpu);
}

DifferentiatePsi::~DifferentiatePsi() noexcept(false) {
    FREE(this->forces, this->gpu)
    FREE(this->O_k_avg, this->gpu)
    FREE(this->E_local_avg, this->gpu)
    FREE(this->E_local_O_k_avg, this->gpu)

    if(this->gpu) {
        FREE(this->x, this->gpu)
        FREE(this->Ax, this->gpu)

        delete[] this->forces_host;
    }
    FREE(this->O_k_avg_x, this->gpu)

    delete this->spin_history_ptr;
    delete this->psi_history_ptr;
}

DifferentiatePsi* network_update;

void evaluate_Ax(
    int*,
    std::complex<double>* x,
    std::complex<double>* Ax
) {
    #ifdef TIMING
    const auto begin = clock::now();
    #endif

    const auto spin_history = network_update->spin_history_ptr->get_kernel();
    const auto psi_history = network_update->psi_history_ptr->get_kernel();
    const auto num_steps = spin_history.get_num_steps();

    if(network_update->gpu) {
        CUDA_CHECK(cudaMemcpy(
            network_update->x,
            x,
            sizeof(std::complex<double>) * network_update->O_k_length,
            cudaMemcpyHostToDevice
        ))
        CUDA_CHECK(cudaMemset(network_update->Ax, 0, sizeof(std::complex<double>) * network_update->O_k_length))
        CUDA_CHECK(cudaMemset(network_update->O_k_avg_x, 0, sizeof(std::complex<double>)))

        const auto blockDim = 256u;

        kernel::DifferentiatePsi network_update_kernel = network_update->get_kernel();

        cuda_kernel<<<num_steps, blockDim>>>(
            [=] __device__ () {
                network_update_kernel.evaluate_first_part_of_Ax(
                    blockIdx.x,
                    spin_history,
                    psi_history
                );
            }
        );
        cuda_kernel<<<network_update->O_k_length / blockDim + 1, blockDim>>>(
            [=] __device__ () {network_update_kernel.compute_O_k_avg_dot_x();}
        );

        cuda_kernel<<<network_update->O_k_length / blockDim + 1, blockDim>>>(
            [=] __device__ () {
                network_update_kernel.finalize_Ax(num_steps);
            }
        );

        CUDA_CHECK(cudaMemcpy(
            Ax,
            network_update->Ax,
            sizeof(std::complex<double>) * network_update->O_k_length,
            cudaMemcpyDeviceToHost
        ))
    }
    else {
        network_update->x = reinterpret_cast<complex_t*>(x);
        network_update->Ax = reinterpret_cast<complex_t*>(Ax);

        for(auto k = 0u; k < network_update->O_k_length; k++) {
            network_update->Ax[k] = complex_t(0.0, 0.0);
        }
        *network_update->O_k_avg_x = complex_t(0.0, 0.0);

        for(auto step = 0u; step < num_steps; step++) {
            network_update->evaluate_first_part_of_Ax(
                step,
                spin_history,
                psi_history
            );
        }
        network_update->compute_O_k_avg_dot_x();
        network_update->finalize_Ax(num_steps);
    }

    #ifdef TIMING
    const auto end = clock::now();
    log_duration("evaluate_Ax", end - begin);
    #endif
}

std::vector<std::complex<double>> DifferentiatePsi::get_O_k_avg() const {
    std::vector<std::complex<double>> result(this->O_k_length);

    const auto nbytes = sizeof(complex_t) * this->O_k_length;
    if (this->gpu) {
        CUDA_CHECK(cudaMemcpy(result.data(), this->O_k_avg, nbytes, cudaMemcpyDeviceToHost))
    }
    else {
        std::memcpy(result.data(), this->O_k_avg, nbytes);
    }

    return result;
}

template<typename SpinEnsemble>
void DifferentiatePsi::operator()(
    const Psi&                  psi,
    const Operator&             operator_,
    const SpinEnsemble&         spin_ensemble,
    std::complex<double>*       result,
    const double                rtol
) {
    const auto blockDim = 256u;

    #ifdef TIMING
    log_init();
    const auto begin_method = clock::now();
    #endif

    if(this->spin_history_ptr == nullptr) {
        this->spin_history_ptr = new SpinHistory(
            spin_ensemble.get_num_steps(), psi.get_num_angles(), spin_ensemble.has_weights(), this->gpu
        );
        this->psi_history_ptr = new PsiHistory(
            spin_ensemble.get_num_steps(), psi.get_num_angles(), this->gpu
        );
    }

    // compute list of spins and angles
    this->spin_history_ptr->fill(psi, spin_ensemble);

    this->num_spins = psi.get_num_spins();
    this->num_hidden_spins = psi.get_num_angles();

    this->psi = psi.get_kernel();

    #ifdef TIMING
    const auto begin_prep = clock::now();
    #endif

    const auto num_steps = this->spin_history_ptr->get_num_steps();
    const auto this_kernel = this->get_kernel();
    const auto operator_kernel = operator_.get_kernel();
    const auto psi_history = this->psi_history_ptr->get_kernel();

    if(this->gpu) {
        CUDA_CHECK(cudaMemset(this->O_k_avg, 0, sizeof(std::complex<double>) * this->O_k_length));
        CUDA_CHECK(cudaMemset(this->E_local_avg, 0, sizeof(std::complex<double>)));
        CUDA_CHECK(cudaMemset(this->E_local_O_k_avg, 0, sizeof(std::complex<double>) * this->O_k_length));

        this->spin_history_ptr->foreach(
            this->psi,
            [=] __device__ (
                const unsigned int step,
                const Spins& spins,
                const complex_t& log_psi,
                const complex_t* angle_ptr,
                const double weight
            ) {
                this_kernel.compute_averages(
                    step, spins, log_psi, angle_ptr, weight, operator_kernel, psi_history
                );
            },
            blockDim
        );

        kernel_divide_array_by_n<<<this->O_k_length / blockDim + 1, blockDim>>>(
            this->O_k_avg, this->O_k_length, 1.0 / num_steps
        );
        kernel_divide_array_by_n<<<this->O_k_length / blockDim + 1, blockDim>>>(
            this->E_local_O_k_avg, this->O_k_length, 1.0 / num_steps
        );
        kernel_divide_array_by_n<<<1, 1>>>(this->E_local_avg, 1, 1.0 / num_steps);

        // compute forces
        cuda_kernel<<<this->O_k_length / blockDim + 1, blockDim>>>(
            [=] __device__ (){this_kernel.compute_forces();}
        );

        // copy forces to host
        CUDA_CHECK(cudaMemcpy(
            this->forces_host,
            this->forces,
            sizeof(complex_t) * this->O_k_length,
            cudaMemcpyDeviceToHost
        ))
    }
    else {
         *this->E_local_avg = complex_t(0.0, 0.0);
        for(auto k = 0u; k < this->O_k_length; k++) {
            this->O_k_avg[k] = complex_t(0.0, 0.0);
            this->E_local_O_k_avg[k] = complex_t(0.0, 0.0);
        }

        this->spin_history_ptr->foreach(
            this->psi,
            [=] __host__ __device__ (
                const unsigned int step,
                const Spins& spins,
                const complex_t& log_psi,
                const complex_t* angle_ptr,
                const double weight
            ) {
                this_kernel.compute_averages(
                    step, spins, log_psi, angle_ptr, weight, operator_kernel, psi_history
                );
            }
        );

        for(auto k = 0u; k < this->O_k_length; k++) {
            this->O_k_avg[k] *= 1.0 / num_steps;
            this->E_local_O_k_avg[k] *= 1.0 / num_steps;
        }
        *this->E_local_avg *= 1.0 / num_steps;

        this->compute_forces();
    }

    #ifdef TIMING
    const auto end_prep = clock::now();
    log_duration("DifferentiatePsi::operator()::preparation", end_prep - begin_prep);
    #endif

    // solve S*x = forces
    network_update = this;
    solve_Ax_b(
        this->O_k_length,
        evaluate_Ax,
        result,
        this->gpu ? this->forces_host : reinterpret_cast<std::complex<double>*>(this->forces),
        rtol
    );

    #ifdef TIMING
    const auto end_method = clock::now();
    log_duration("DifferentiatePsi::operator()", end_method - begin_method);
    log_flush();
    #endif
}


template void DifferentiatePsi::operator()(
    const Psi&                  psi,
    const Operator&             operator_,
    const ExactSummation&       spin_ensemble,
    std::complex<double>*       result,
    const double                rtol
);
template void DifferentiatePsi::operator()(
    const Psi&                  psi,
    const Operator&             operator_,
    const MonteCarloLoop&       spin_ensemble,
    std::complex<double>*       result,
    const double                rtol
);

} // namespace rbm_on_gpu
