#define __PYTHONCC__
#include "quantum_state/Psi.hpp"
#include "quantum_state/PsiW3.hpp"
#include "quantum_state/PsiDynamical.hpp"
#include "operator/Operator.hpp"
#include "spin_ensembles/ExactSummation.hpp"
#include "spin_ensembles/MonteCarloLoop.hpp"
#include "spin_ensembles/SpinHistory.hpp"
#include "network_functions/ExpectationValue.hpp"
#include "network_functions/DifferentiatePsi.hpp"
#include "network_functions/HilbertSpaceDistance.hpp"
#include "network_functions/HilbertSpaceDistanceV2.hpp"
#include "network_functions/PsiOkVector.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pytensor.hpp>

#include <iostream>
#include <complex>


namespace py = pybind11;

using namespace rbm_on_gpu;
using namespace pybind11::literals;

template<unsigned int dim>
using complex_tensor = xt::pytensor<std::complex<double>, dim>;

// Python Module and Docstrings

PYBIND11_MODULE(_pyRBMonGPU, m)
{
    xt::import_numpy();

    py::class_<Psi>(m, "Psi")
        .def(py::init<
            const complex_tensor<1u>&,
            const complex_tensor<1u>&,
            const complex_tensor<2u>&,
            const complex_tensor<1u>&,
            const double,
            const bool
        >())
        .def(
            "update_params",
            py::overload_cast<
                const complex_tensor<1u>&,
                const complex_tensor<1u>&,
                const complex_tensor<2u>&,
                const complex_tensor<1u>&
            >(&Psi::update_params)
        )
        .def_property_readonly("vector", &Psi::as_vector_py)
        .def_property_readonly("norm", &Psi::norm_function)
        .def("psi_s", &Psi::psi_s_std)
        .def("O_k_vector", &Psi::O_k_vector_py)
        .def("log_psi", &Psi::log_psi_s_std)
        .def("probability_s", &Psi::probability_s_py)
        .def_readwrite("prefactor", &Psi::prefactor)
        .def_readonly("gpu", &Psi::gpu)
        .def_readonly("N", &Psi::N)
        .def_readonly("M", &Psi::M)
        .def_property_readonly("num_params", &Psi::get_num_params_py)
        .def_readonly("num_active_params", &Psi::num_active_params)
        .def_property_readonly("num_angles", &Psi::get_num_angles);

    py::class_<PsiW3>(m, "PsiW3")
        .def(py::init<
            const complex_tensor<1u>&,
            const complex_tensor<1u>&,
            const complex_tensor<2u>&,
            const complex_tensor<1u>&,
            const complex_tensor<2u>&,
            const complex_tensor<2u>&,
            const double,
            const bool
        >())
        .def(
            "update_params",
            py::overload_cast<
                const complex_tensor<1u>&,
                const complex_tensor<1u>&,
                const complex_tensor<2u>&,
                const complex_tensor<1u>&,
                const complex_tensor<2u>&,
                const complex_tensor<2u>&
            >(&PsiW3::update_params)
        )
        .def_property_readonly("vector", &PsiW3::as_vector_py)
        .def_property_readonly("norm", &PsiW3::norm_function)
        .def("O_k_vector", &PsiW3::O_k_vector_py)
        .def_readwrite("prefactor", &PsiW3::prefactor)
        .def_readonly("gpu", &PsiW3::gpu)
        .def_readonly("N", &PsiW3::N)
        .def_readonly("M", &PsiW3::M)
        .def_readonly("F", &PsiW3::F)
        .def_property_readonly("num_params", &PsiW3::get_num_params_py)
        .def_readonly("num_active_params", &PsiW3::num_active_params);

    py::class_<PsiDynamical::Link>(m, "PsiDynamical_Link")
        .def_readonly("first_spin", &PsiDynamical::Link::first_spin)
        .def_readonly("weights", &PsiDynamical::Link::weights)
        .def_readonly("hidden_spin_weight", &PsiDynamical::Link::hidden_spin_weight)
        .def_readonly("hidden_spin_type", &PsiDynamical::Link::hidden_spin_type);

    py::class_<PsiDynamical>(m, "PsiDynamical")
        .def(py::init<vector<complex<double>>, const bool>())
        .def("copy", &PsiDynamical::copy)
        .def("add_hidden_spin", &PsiDynamical::add_hidden_spin)
        .def(
            "update", &PsiDynamical::update, "resize"_a = true
        )
        .def_property_readonly("vector", &PsiDynamical::as_vector_py)
        .def_property_readonly("norm", &PsiDynamical::norm_function)
        .def("O_k_vector", &PsiDynamical::O_k_vector_py)
        .def_readwrite("prefactor", &PsiDynamical::prefactor)
        .def_readonly("gpu", &PsiDynamical::gpu)
        .def_readonly("N", &PsiDynamical::N)
        .def_readonly("M", &PsiDynamical::M)
        .def_readonly("spin_weights", &PsiDynamical::spin_weights)
        .def_property_readonly("num_params", &PsiDynamical::get_num_params_py)
        .def_readonly("num_active_params", &PsiDynamical::num_active_params)
        .def_property("active_params", &PsiDynamical::get_active_params_py, &PsiDynamical::set_active_params_py)
        .def_property_readonly("active_params_types", &PsiDynamical::get_active_params_types_py)
        .def_readonly("links", &PsiDynamical::links)
        .def_property("a", &PsiDynamical::a_py, &PsiDynamical::set_a_py)
        .def_property_readonly("b", &PsiDynamical::b_py)
        .def_property_readonly("W", &PsiDynamical::dense_W_py)
        .def_property_readonly("num_angles", &PsiDynamical::get_num_angles);

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
        .def("flip", &rbm_on_gpu::Spins::flip);

    py::class_<MonteCarloLoop>(m, "MonteCarloLoop")
        .def(py::init<unsigned int, unsigned int, unsigned int, unsigned int, bool>())
        .def(py::init<const MonteCarloLoop&>());

    py::class_<ExactSummation>(m, "ExactSummation")
        .def(py::init<unsigned int>())
        .def(py::init<const Psi&>())
        .def(py::init<const PsiW3&>())
        .def("copy", &ExactSummation::copy);

    py::class_<SpinHistory>(m, "SpinHistory")
        .def(py::init<const unsigned int, const unsigned int, const bool, const bool>())
        .def("fill", &SpinHistory::fill<Psi, ExactSummation>)
        .def("fill", &SpinHistory::fill<Psi, MonteCarloLoop>)
        .def("fill", &SpinHistory::fill<PsiDynamical, ExactSummation>)
        .def("fill", &SpinHistory::fill<PsiDynamical, MonteCarloLoop>)
        .def("toggle_angles", &SpinHistory::toggle_angles);

    py::class_<ExpectationValue>(m, "ExpectationValue")
        .def(py::init<bool>())
        .def("__call__", &ExpectationValue::__call__<Psi, ExactSummation>)
        .def("__call__", &ExpectationValue::__call__vector<Psi, ExactSummation>)
        .def("__call__", &ExpectationValue::__call__<Psi, MonteCarloLoop>)
        .def("__call__", &ExpectationValue::__call__vector<Psi, MonteCarloLoop>)
        .def("__call__", &ExpectationValue::__call__<PsiW3, ExactSummation>)
        .def("__call__", &ExpectationValue::__call__vector<PsiW3, ExactSummation>)
        .def("__call__", &ExpectationValue::__call__<PsiW3, MonteCarloLoop>)
        .def("__call__", &ExpectationValue::__call__vector<PsiW3, MonteCarloLoop>)
        .def("__call__", &ExpectationValue::__call__<PsiDynamical, ExactSummation>)
        .def("__call__", &ExpectationValue::__call__vector<PsiDynamical, ExactSummation>)
        .def("__call__", &ExpectationValue::__call__<PsiDynamical, MonteCarloLoop>)
        .def("__call__", &ExpectationValue::__call__vector<PsiDynamical, MonteCarloLoop>)
        .def("fluctuation", &ExpectationValue::fluctuation<Psi, ExactSummation>)
        .def("fluctuation", &ExpectationValue::fluctuation<Psi, MonteCarloLoop>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiDynamical, ExactSummation>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiDynamical, MonteCarloLoop>)
        .def("gradient", &ExpectationValue::gradient_py<Psi, ExactSummation>)
        .def("gradient", &ExpectationValue::gradient_py<Psi, MonteCarloLoop>)
        .def("gradient", &ExpectationValue::gradient_py<PsiDynamical, ExactSummation>)
        .def("gradient", &ExpectationValue::gradient_py<PsiDynamical, MonteCarloLoop>)
        .def("fluctuation_gradient", &ExpectationValue::fluctuation_gradient_py<Psi, ExactSummation>)
        .def("fluctuation_gradient", &ExpectationValue::fluctuation_gradient_py<Psi, MonteCarloLoop>)
        .def("fluctuation_gradient", &ExpectationValue::fluctuation_gradient_py<PsiDynamical, ExactSummation>)
        .def("fluctuation_gradient", &ExpectationValue::fluctuation_gradient_py<PsiDynamical, MonteCarloLoop>)
        .def("difference", &ExpectationValue::difference<Psi, ExactSummation>)
        .def("difference", &ExpectationValue::difference<Psi, MonteCarloLoop>)
        .def("difference", &ExpectationValue::difference<PsiDynamical, ExactSummation>)
        .def("difference", &ExpectationValue::difference<PsiDynamical, MonteCarloLoop>);

    py::class_<DifferentiatePsi>(m, "DifferentiatePsi")
        .def(py::init<unsigned int, bool>())
        .def("__call__", &DifferentiatePsi::__call__<ExactSummation>)
        .def("__call__", &DifferentiatePsi::__call__<MonteCarloLoop>)
        .def("get_O_k_avg", &DifferentiatePsi::get_O_k_avg);

    py::class_<HilbertSpaceDistance>(m, "HilbertSpaceDistance")
        .def(py::init<unsigned int, bool>())
        .def("__call__", &HilbertSpaceDistance::distance<Psi, ExactSummation>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "use_record"_a = false)
        .def("__call__", &HilbertSpaceDistance::distance<Psi, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "use_record"_a = false)
        .def("__call__", &HilbertSpaceDistance::distance<Psi, SpinHistory>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "use_record"_a = false)
        .def("__call__", &HilbertSpaceDistance::distance<PsiDynamical, ExactSummation>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "use_record"_a = false)
        .def("__call__", &HilbertSpaceDistance::distance<PsiDynamical, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "use_record"_a = false)
        .def("__call__", &HilbertSpaceDistance::distance<PsiDynamical, SpinHistory>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "use_record"_a = false)
        .def("gradient", &HilbertSpaceDistance::gradient_py<Psi, ExactSummation>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "use_record"_a = false)
        .def("gradient", &HilbertSpaceDistance::gradient_py<Psi, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "use_record"_a = false)
        .def("gradient", &HilbertSpaceDistance::gradient_py<Psi, SpinHistory>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "use_record"_a = false)
        .def("gradient", &HilbertSpaceDistance::gradient_py<PsiDynamical, ExactSummation>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "use_record"_a = false)
        .def("gradient", &HilbertSpaceDistance::gradient_py<PsiDynamical, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "use_record"_a = false)
        .def("gradient", &HilbertSpaceDistance::gradient_py<PsiDynamical, SpinHistory>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "use_record"_a = false)
        .def("record", &HilbertSpaceDistance::record<PsiDynamical, SpinHistory>);

    py::class_<HilbertSpaceDistanceV2>(m, "HilbertSpaceDistanceV2")
        .def(py::init<bool>())
        .def("__call__", &HilbertSpaceDistanceV2::distance<ExactSummation>)
        .def("__call__", &HilbertSpaceDistanceV2::distance<MonteCarloLoop>);

    m.def("psi_O_k_vector", psi_O_k_vector_py<Psi, ExactSummation>);
    m.def("psi_O_k_vector", psi_O_k_vector_py<Psi, MonteCarloLoop>);
    m.def("psi_O_k_vector", psi_O_k_vector_py<PsiDynamical, ExactSummation>);
    m.def("psi_O_k_vector", psi_O_k_vector_py<PsiDynamical, MonteCarloLoop>);

    m.def("setDevice", setDevice);
}
