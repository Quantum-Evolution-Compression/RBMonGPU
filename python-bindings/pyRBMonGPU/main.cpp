#define __PYTHONCC__
#include "quantum_state/PsiClassical.hpp"
#include "quantum_state/Psi.hpp"
#include "quantum_state/PsiDeep.hpp"
#include "quantum_state/PsiDeepMin.hpp"
#include "quantum_state/PsiHamiltonian.hpp"
#include "operator/Operator.hpp"
#include "spin_ensembles/ExactSummation.hpp"
#include "spin_ensembles/MonteCarloLoop.hpp"
#include "network_functions/ExpectationValue.hpp"
#include "network_functions/HilbertSpaceDistance.hpp"
#include "network_functions/PsiOkVector.hpp"
#include "network_functions/PsiAngles.hpp"
#include "network_functions/S_matrix.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY
#include "xtensor/xadapt.hpp"
#include <xtensor-python/pytensor.hpp>

#include <iostream>
#include <complex>


namespace py = pybind11;

using namespace rbm_on_gpu;
using namespace pybind11::literals;

template<unsigned int dim>
using complex_tensor = xt::pytensor<std::complex<double>, dim>;

template<unsigned int dim>
using real_tensor = xt::pytensor<double, dim>;

// Python Module and Docstrings

PYBIND11_MODULE(_pyRBMonGPU, m)
{
    xt::import_numpy();

    py::class_<Psi>(m, "Psi")
        .def(py::init<
            const real_tensor<1u>&,
            const real_tensor<1u>&,
            const complex_tensor<1u>&,
            const complex_tensor<2u>&,
            const double,
            const bool,
            const bool
        >())
        .def("copy", &Psi::copy)
        .def_property_readonly("_vector", &Psi::as_vector_py)
        .def("norm", &Psi::norm_function)
        .def("O_k_vector", &Psi::O_k_vector_py)
        .def_readwrite("prefactor", &Psi::prefactor)
        .def_readonly("gpu", &Psi::gpu)
        .def_readonly("N", &Psi::N)
        .def_readonly("M", &Psi::M)
        .def_property(
            "alpha",
            [](const Psi& psi){return psi.alpha_array.to_pytensor<1u>();},
            [](Psi& psi, const real_tensor<1u>& input) {psi.alpha_array = input;}
        )
        .def_property(
            "beta",
            [](const Psi& psi){return psi.beta_array.to_pytensor<1u>();},
            [](Psi& psi, const real_tensor<1u>& input) {psi.beta_array = input;}
        )
        .def_property(
            "b",
            [](const Psi& psi){return psi.b_array.to_pytensor<1u>();},
            [](Psi& psi, const complex_tensor<1u>& input) {psi.b_array = input; psi.update_kernel();}
        )
        .def_property(
            "W",
            [](const Psi& psi){return psi.W_array.to_pytensor<2u>(shape_t<2u>{psi.N, psi.M});},
            [](Psi& psi, const complex_tensor<2u>& input) {psi.W_array = input; psi.update_kernel();}
        )
        .def_readonly("num_params", &Psi::num_params)
        .def_property("params", &Psi::get_params_py, &Psi::set_params_py)
        .def_property_readonly("free_quantum_axis", [](const Psi& psi) {return psi.free_quantum_axis;})
        .def_property_readonly("num_angles", &Psi::get_num_angles);

    py::class_<PsiDeep>(m, "PsiDeep")
        .def(py::init<
            const real_tensor<1u>&,
            const real_tensor<1u>&,
            const vector<complex_tensor<1u>>&,
            const vector<xt::pytensor<unsigned int, 2u>>&,
            const vector<complex_tensor<2u>>&,
            const double,
            const bool,
            const bool
        >())
        .def("copy", &PsiDeep::copy)
        .def_readwrite("prefactor", &PsiDeep::prefactor)
        .def_readwrite("N_i", &PsiDeep::N_i)
        .def_readwrite("N_j", &PsiDeep::N_j)
        .def_readonly("gpu", &PsiDeep::gpu)
        .def_readonly("N", &PsiDeep::N)
        .def_readonly("num_params", &PsiDeep::num_params)
        .def_property(
            "params",
            [](const PsiDeep& psi) {return psi.get_params().to_pytensor<1u>();},
            [](PsiDeep& psi, const complex_tensor<1u>& new_params) {psi.set_params(Array<complex_t>(new_params, false));}
        )
        .def_property_readonly("alpha", [](const PsiDeep& psi) {return psi.alpha_array.to_pytensor<1u>();})
        .def_property_readonly("beta", [](const PsiDeep& psi) {return psi.beta_array.to_pytensor<1u>();})
        .def_property_readonly("b", &PsiDeep::get_b)
        .def_property_readonly("connections", &PsiDeep::get_connections)
        .def_property_readonly("W", &PsiDeep::get_W)
        .def_property_readonly("_vector", [](const PsiDeep& psi) {return psi.as_vector().to_pytensor<1u>();})
        .def_property_readonly("free_quantum_axis", [](const PsiDeep& psi) {return psi.free_quantum_axis;})
        .def("norm", &PsiDeep::norm)
        .def("O_k_vector", &PsiDeep::O_k_vector_py);

    py::class_<PsiClassical>(m, "PsiClassical")
        .def(py::init<
            const unsigned int,
            const unsigned int,
            const bool
        >())
        .def_readonly("gpu", &PsiClassical::gpu)
        .def_readonly("N", &PsiClassical::N)
        // .def_property(
        //     "W",
        //     [](const PsiClassical& psi){return psi.W_array.to_pytensor<2u>(shape_t<2u>{psi.N, psi.M});},
        //     [](PsiClassical& psi, const complex_tensor<2u>& input) {psi.W_array = input; psi.update_kernel();}
        //
        // .def("log_psi_s", &PsiClassical::log_psi_s)
        .def_property_readonly("vector", [](const PsiClassical& psi) {return psi_vector(psi).to_pytensor<1u>();})
        // .def("log_psi_s", &PsiClassical::log_psi_s)
        .def_readonly("num_params", &PsiClassical::num_params)
        .def_property_readonly("free_quantum_axis", [](const PsiClassical& psi) {return psi.free_quantum_axis;})
        .def_property_readonly("num_angles", &PsiClassical::get_num_angles);

    py::class_<PsiDeepMin>(m, "PsiDeepMin")
        .def(py::init<
            const string
        >())
        .def_readonly("N", &PsiDeepMin::N)
        // .def_property_readonly("vector", [](const PsiClassical& psi) {return psi_vector(psi).to_pytensor<1u>();})
        .def("log_psi_s", &PsiDeepMin::log_psi_s);

    py::class_<PsiHamiltonian>(m, "PsiHamiltonian")
        .def(py::init<
            const unsigned int,
            const Operator&
        >())
        .def_readonly("gpu", &PsiHamiltonian::gpu)
        .def_readonly("N", &PsiHamiltonian::N);

    py::class_<Operator>(m, "Operator")
        .def(py::init<
            const complex_tensor<1u>&,
            const xt::pytensor<int, 2u>&,
            const xt::pytensor<int, 2u>&,
            const bool
        >())
        .def(py::init<
            const quantum_expression::PauliExpression&,
            const bool
        >())
        .def_property_readonly("expr", &Operator::to_expr)
        .def_readonly("gpu", &Operator::gpu)
        .def_readonly("num_strings", &Operator::num_strings)
        .def_readonly("max_string_length", &Operator::max_string_length)
        .def_property_readonly("coefficients", &Operator::get_coefficients_py)
        .def_property_readonly("pauli_types", &Operator::get_pauli_types_py)
        .def_property_readonly("pauli_indices", &Operator::get_pauli_indices_py);

    py::class_<rbm_on_gpu::Spins>(m, "Spins")
        .def(py::init<rbm_on_gpu::Spins::type>())
        .def("array", &rbm_on_gpu::Spins::array)
        .def("flip", &rbm_on_gpu::Spins::flip)
        .def("rotate_left", &rbm_on_gpu::Spins::rotate_left)
        .def("shift_2d", &rbm_on_gpu::Spins::shift_2d);

    py::class_<MonteCarloLoop>(m, "MonteCarloLoop")
        .def(py::init<unsigned int, unsigned int, unsigned int, unsigned int, bool>())
        .def(py::init<const MonteCarloLoop&>())
        .def("set_total_z_symmetry", &MonteCarloLoop::set_total_z_symmetry)
        .def_property_readonly("num_steps", &MonteCarloLoop::get_num_steps);

    py::class_<ExactSummation>(m, "ExactSummation")
        .def(py::init<unsigned int, bool>())
        .def("set_total_z_symmetry", &ExactSummation::set_total_z_symmetry)
        .def_property_readonly("num_steps", &ExactSummation::get_num_steps);

    py::class_<ExpectationValue>(m, "ExpectationValue")
        .def(py::init<bool>())
        .def("__call__", &ExpectationValue::__call__<Psi, ExactSummation>)
        .def("__call__", &ExpectationValue::__call__vector<Psi, ExactSummation>)
        .def("__call__", &ExpectationValue::__call__<Psi, MonteCarloLoop>)
        .def("__call__", &ExpectationValue::__call__vector<Psi, MonteCarloLoop>)
        .def("__call__", &ExpectationValue::__call__<PsiDeep, ExactSummation>)
        .def("__call__", &ExpectationValue::__call__vector<PsiDeep, ExactSummation>)
        .def("__call__", &ExpectationValue::__call__<PsiDeep, MonteCarloLoop>)
        .def("__call__", &ExpectationValue::__call__vector<PsiDeep, MonteCarloLoop>)
        .def("__call__", &ExpectationValue::__call__<PsiHamiltonian, MonteCarloLoop>)
        .def("fluctuation", &ExpectationValue::fluctuation<Psi, ExactSummation>)
        .def("fluctuation", &ExpectationValue::fluctuation<Psi, MonteCarloLoop>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiDeep, ExactSummation>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiDeep, MonteCarloLoop>)
        .def("gradient", &ExpectationValue::gradient_py<Psi, ExactSummation>)
        .def("gradient", &ExpectationValue::gradient_py<Psi, MonteCarloLoop>)
        .def("gradient", &ExpectationValue::gradient_py<PsiDeep, ExactSummation>)
        .def("gradient", &ExpectationValue::gradient_py<PsiDeep, MonteCarloLoop>)
        .def("fluctuation_gradient", &ExpectationValue::fluctuation_gradient_py<Psi, ExactSummation>)
        .def("fluctuation_gradient", &ExpectationValue::fluctuation_gradient_py<Psi, MonteCarloLoop>)
        .def("fluctuation_gradient", &ExpectationValue::fluctuation_gradient_py<PsiDeep, ExactSummation>)
        .def("fluctuation_gradient", &ExpectationValue::fluctuation_gradient_py<PsiDeep, MonteCarloLoop>)
        .def("difference", &ExpectationValue::difference<Psi, ExactSummation>)
        .def("difference", &ExpectationValue::difference<Psi, MonteCarloLoop>)
        .def("difference", &ExpectationValue::difference<PsiDeep, ExactSummation>)
        .def("difference", &ExpectationValue::difference<PsiDeep, MonteCarloLoop>);

    py::class_<HilbertSpaceDistance>(m, "HilbertSpaceDistance")
        .def(py::init<unsigned int, unsigned int, bool>())
        .def("__call__", &HilbertSpaceDistance::distance<Psi, Psi, ExactSummation>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("__call__", &HilbertSpaceDistance::distance<Psi, Psi, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("__call__", &HilbertSpaceDistance::distance<PsiDeep, PsiDeep, ExactSummation>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("__call__", &HilbertSpaceDistance::distance<PsiDeep, PsiDeep, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        // .def("overlap", &HilbertSpaceDistance::overlap<Psi, ExactSummation>, "psi"_a, "psi_prime"_a, "spin_ensemble"_a)
        // .def("overlap", &HilbertSpaceDistance::overlap<Psi, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "spin_ensemble"_a)
        // .def("overlap", &HilbertSpaceDistance::overlap<PsiDeep, ExactSummation>, "psi"_a, "psi_prime"_a, "spin_ensemble"_a)
        // .def("overlap", &HilbertSpaceDistance::overlap<PsiDeep, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<Psi, Psi, ExactSummation>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<Psi, Psi, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<PsiDeep, PsiDeep, ExactSummation>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<PsiDeep, PsiDeep, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("__call__", &HilbertSpaceDistance::distance<PsiClassical, PsiDeep, ExactSummation>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<PsiClassical, PsiDeep, ExactSummation>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<PsiClassical, PsiDeep, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a);;

    m.def("get_S_matrix", [](const Psi& psi, const ExactSummation& spin_ensemble){
        return get_S_matrix(psi, spin_ensemble).to_pytensor<2u>(shape_t<2u>{psi.num_params, psi.num_params});
    });

    m.def("get_O_k_vector", [](const Psi& psi, const ExactSummation& spin_ensemble) {
        auto result_and_result_std = psi_O_k_vector(psi, spin_ensemble);
        return make_pair(
            result_and_result_std.first.to_pytensor<1u>(),
            result_and_result_std.second.to_pytensor<1u>()
        );
    });
    m.def("get_O_k_vector", [](const Psi& psi, const MonteCarloLoop& spin_ensemble) {
        auto result_and_result_std = psi_O_k_vector(psi, spin_ensemble);
        return make_pair(
            result_and_result_std.first.to_pytensor<1u>(),
            result_and_result_std.second.to_pytensor<1u>()
        );
    });
    m.def("get_O_k_vector", [](const PsiDeep& psi, const ExactSummation& spin_ensemble) {
        auto result_and_result_std = psi_O_k_vector(psi, spin_ensemble);
        return make_pair(
            result_and_result_std.first.to_pytensor<1u>(),
            result_and_result_std.second.to_pytensor<1u>()
        );
    });
    m.def("get_O_k_vector", [](const PsiDeep& psi, const MonteCarloLoop& spin_ensemble) {
        auto result_and_result_std = psi_O_k_vector(psi, spin_ensemble);
        return make_pair(
            result_and_result_std.first.to_pytensor<1u>(),
            result_and_result_std.second.to_pytensor<1u>()
        );
    });

    m.def("psi_angles", [](const PsiDeep& psi, const ExactSummation& spin_ensemble) {
        auto result_and_result_std = psi_angles(psi, spin_ensemble);
        return make_pair(
            result_and_result_std.first.to_pytensor<1u>(),
            result_and_result_std.second.to_pytensor<1u>()
        );
    });
    m.def("psi_angles", [](const PsiDeep& psi, const MonteCarloLoop& spin_ensemble) {
        auto result_and_result_std = psi_angles(psi, spin_ensemble);
        return make_pair(
            result_and_result_std.first.to_pytensor<1u>(),
            result_and_result_std.second.to_pytensor<1u>()
        );
    });

    m.def("activation_function", [](const complex<double>& x) {
        return my_logcosh(complex_t(x.real(), x.imag())).to_std();
    });

    m.def("setDevice", setDevice);
    m.def("start_profiling", start_profiling);
    m.def("stop_profiling", stop_profiling);
}
