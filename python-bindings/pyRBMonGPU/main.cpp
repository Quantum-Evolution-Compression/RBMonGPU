#define __PYTHONCC__
#include "quantum_state/Psi.hpp"
#include "quantum_state/PsiDynamical.hpp"
#include "quantum_state/PsiDeep.hpp"
#include "operator/Operator.hpp"
#include "spin_ensembles/ExactSummation.hpp"
#include "spin_ensembles/MonteCarloLoop.hpp"
#include "network_functions/ExpectationValue.hpp"
#include "network_functions/HilbertSpaceDistance.hpp"
#include "network_functions/PsiOkVector.hpp"

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
using complex_tensor = xt::pytensor<std::complex<float>, dim>;

// Python Module and Docstrings

PYBIND11_MODULE(_pyRBMonGPU, m)
{
    xt::import_numpy();

    py::class_<Psi>(m, "Psi")
        .def(py::init<
            const complex_tensor<1u>&,
            const complex_tensor<1u>&,
            const complex_tensor<2u>&,
            const float,
            const bool
        >())
        .def("copy", &Psi::copy)
        .def_property_readonly("vector", &Psi::as_vector_py)
        .def("norm", &Psi::norm_function)
        .def("O_k_vector", &Psi::O_k_vector_py)
        .def_readwrite("prefactor", &Psi::prefactor)
        .def_readonly("gpu", &Psi::gpu)
        .def_readonly("N", &Psi::N)
        .def_readonly("M", &Psi::M)
        .def_property(
            "a",
            [](const Psi& psi){return psi.a_array.to_pytensor<1u>();},
            [](Psi& psi, const complex_tensor<1u>& input) {psi.a_array = input; psi.update_kernel();}
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
        .def_property_readonly("num_angles", &Psi::get_num_angles)
        .def_readonly("index_pairs", &Psi::index_pair_list);

    py::class_<PsiDynamical::Link>(m, "PsiDynamical_Link")
        .def_readonly("first_spin", &PsiDynamical::Link::first_spin)
        .def_readonly("weights", &PsiDynamical::Link::weights)
        .def_readonly("hidden_spin_weight", &PsiDynamical::Link::hidden_spin_weight);

    py::class_<PsiDynamical>(m, "PsiDynamical")
        .def(py::init<vector<complex<float>>, const bool>())
        .def("copy", &PsiDynamical::copy)
        .def("add_hidden_spin", &PsiDynamical::add_hidden_spin)
        .def(
            "update", &PsiDynamical::update, "resize"_a = true
        )
        .def_property_readonly("vector", &PsiDynamical::as_vector_py)
        .def("norm", &PsiDynamical::norm_function)
        .def("O_k_vector", &PsiDynamical::O_k_vector_py)
        .def_readwrite("prefactor", &PsiDynamical::prefactor)
        .def_readonly("gpu", &PsiDynamical::gpu)
        .def_readonly("N", &PsiDynamical::N)
        .def_readonly("M", &PsiDynamical::M)
        .def_readonly("spin_weights", &PsiDynamical::spin_weights)
        .def_property_readonly("num_params", &PsiDynamical::get_num_params_py)
        .def_readonly("num_params", &PsiDynamical::num_params)
        .def_property("params", &PsiDynamical::get_params_py, &PsiDynamical::set_params_py)
        .def_readonly("links", &PsiDynamical::links)
        .def_property("a", &PsiDynamical::a_py, &PsiDynamical::set_a_py)
        .def_property_readonly("b", &PsiDynamical::b_py)
        .def_property_readonly("W", &PsiDynamical::dense_W_py)
        .def_property_readonly("num_angles", &PsiDynamical::get_num_angles)
        .def_readonly("index_pairs", &PsiDynamical::index_pair_list);

    py::class_<PsiDeep>(m, "PsiDeep")
        .def(py::init<
            const complex_tensor<1u>&,
            const vector<complex_tensor<1u>>&,
            const vector<complex_tensor<2u>>&,
            const float,
            const bool
        >())
        .def("copy", &PsiDeep::copy)
        .def_readwrite("prefactor", &PsiDeep::prefactor)
        .def_readonly("gpu", &PsiDeep::gpu)
        .def_readonly("N", &PsiDeep::N)
        .def_readonly("num_params", &PsiDeep::num_params)
        .def_property(
            "params",
            [](const PsiDeep& psi) {return psi.get_params().to_pytensor<1u>();},
            [](PsiDeep& psi, const complex_tensor<1u>& new_params) {psi.set_params(Array<complex_t>(new_params, false));}
        )
        .def_property_readonly("a", [](const PsiDeep& psi) {return psi.a_array.to_pytensor<1u>();})
        .def_property_readonly("b", &PsiDeep::get_b)
        .def_property_readonly("W", &PsiDeep::get_W)
        .def_property_readonly("vector", [](const PsiDeep& psi) {return psi.as_vector().to_pytensor<1u>();})
        .def("norm", &PsiDeep::norm)
        .def("O_k_vector", &PsiDeep::O_k_vector_py);

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
        .def(py::init<const MonteCarloLoop&>())
        .def("set_total_z_symmetry", &MonteCarloLoop::set_total_z_symmetry)
        .def_property_readonly("num_steps", &MonteCarloLoop::get_num_steps);

    py::class_<ExactSummation>(m, "ExactSummation")
        .def(py::init<unsigned int, bool>())
        .def("set_total_z_symmetry", &ExactSummation::set_total_z_symmetry)
        .def_property_readonly("num_steps", &ExactSummation::get_num_steps);
/*
    py::class_<ExpectationValue>(m, "ExpectationValue")
        .def(py::init<bool>())
        .def("__call__", &ExpectationValue::__call__<Psi, ExactSummation>)
        .def("__call__", &ExpectationValue::__call__vector<Psi, ExactSummation>)
        .def("__call__", &ExpectationValue::__call__<Psi, MonteCarloLoop>)
        .def("__call__", &ExpectationValue::__call__vector<Psi, MonteCarloLoop>)
        .def("__call__", &ExpectationValue::__call__<PsiDynamical, ExactSummation>)
        .def("__call__", &ExpectationValue::__call__vector<PsiDynamical, ExactSummation>)
        .def("__call__", &ExpectationValue::__call__<PsiDynamical, MonteCarloLoop>)
        .def("__call__", &ExpectationValue::__call__vector<PsiDynamical, MonteCarloLoop>)
        .def("__call__", &ExpectationValue::__call__<PsiDeep, ExactSummation>)
        .def("__call__", &ExpectationValue::__call__vector<PsiDeep, ExactSummation>)
        .def("__call__", &ExpectationValue::__call__<PsiDeep, MonteCarloLoop>)
        .def("__call__", &ExpectationValue::__call__vector<PsiDeep, MonteCarloLoop>)
        .def("fluctuation", &ExpectationValue::fluctuation<Psi, ExactSummation>)
        .def("fluctuation", &ExpectationValue::fluctuation<Psi, MonteCarloLoop>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiDynamical, ExactSummation>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiDynamical, MonteCarloLoop>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiDeep, ExactSummation>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiDeep, MonteCarloLoop>)
        .def("gradient", &ExpectationValue::gradient_py<Psi, ExactSummation>)
        .def("gradient", &ExpectationValue::gradient_py<Psi, MonteCarloLoop>)
        .def("gradient", &ExpectationValue::gradient_py<PsiDynamical, ExactSummation>)
        .def("gradient", &ExpectationValue::gradient_py<PsiDynamical, MonteCarloLoop>)
        .def("gradient", &ExpectationValue::gradient_py<PsiDeep, ExactSummation>)
        .def("gradient", &ExpectationValue::gradient_py<PsiDeep, MonteCarloLoop>)
        .def("fluctuation_gradient", &ExpectationValue::fluctuation_gradient_py<Psi, ExactSummation>)
        .def("fluctuation_gradient", &ExpectationValue::fluctuation_gradient_py<Psi, MonteCarloLoop>)
        .def("fluctuation_gradient", &ExpectationValue::fluctuation_gradient_py<PsiDynamical, ExactSummation>)
        .def("fluctuation_gradient", &ExpectationValue::fluctuation_gradient_py<PsiDynamical, MonteCarloLoop>)
        .def("fluctuation_gradient", &ExpectationValue::fluctuation_gradient_py<PsiDeep, ExactSummation>)
        .def("fluctuation_gradient", &ExpectationValue::fluctuation_gradient_py<PsiDeep, MonteCarloLoop>)
        .def("difference", &ExpectationValue::difference<Psi, ExactSummation>)
        .def("difference", &ExpectationValue::difference<Psi, MonteCarloLoop>)
        .def("difference", &ExpectationValue::difference<PsiDynamical, ExactSummation>)
        .def("difference", &ExpectationValue::difference<PsiDynamical, MonteCarloLoop>)
        .def("difference", &ExpectationValue::difference<PsiDeep, ExactSummation>)
        .def("difference", &ExpectationValue::difference<PsiDeep, MonteCarloLoop>);

    py::class_<HilbertSpaceDistance>(m, "HilbertSpaceDistance")
        .def(py::init<unsigned int, bool>())
        .def("__call__", &HilbertSpaceDistance::distance<Psi, ExactSummation>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("__call__", &HilbertSpaceDistance::distance<Psi, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("__call__", &HilbertSpaceDistance::distance<PsiDynamical, ExactSummation>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("__call__", &HilbertSpaceDistance::distance<PsiDynamical, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("__call__", &HilbertSpaceDistance::distance<PsiDeep, ExactSummation>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("__call__", &HilbertSpaceDistance::distance<PsiDeep, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<Psi, ExactSummation>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<Psi, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<PsiDynamical, ExactSummation>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<PsiDynamical, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<PsiDeep, ExactSummation>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<PsiDeep, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a);
*/
    m.def("setDevice", setDevice);
}
