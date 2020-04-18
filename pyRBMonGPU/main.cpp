#define __PYTHONCC__
#include "quantum_states.hpp"
#include "spin_ensembles.hpp"
#include "operator/Operator.hpp"
#include "network_functions/ExpectationValue.hpp"
#include "network_functions/HilbertSpaceDistance.hpp"
#include "network_functions/KullbackLeibler.hpp"
#include "network_functions/PsiVector.hpp"
#include "network_functions/PsiNorm.hpp"
#include "network_functions/PsiOkVector.hpp"
#include "network_functions/PsiAngles.hpp"
#include "network_functions/S_matrix.hpp"
#include "network_functions/RenyiCorrelation.hpp"
#include "types.h"

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

#ifdef ENABLE_PSI
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
    #ifdef ENABLE_EXACT_SUMMATION
        .def_property_readonly("vector", [](const Psi& psi) {return psi_vector_py(psi);})
        .def("norm", [](const Psi& psi, ExactSummation& exact_summation) {return psi_norm(psi, exact_summation);})
    #endif // ENABLE_EXACT_SUMMATION
        .def_readwrite("prefactor", &Psi::prefactor)
        .def_readonly("gpu", &Psi::gpu)
        .def_readonly("N", &Psi::N)
        .def_readonly("M", &Psi::M)
        .def_property(
            "alpha",
            [](const Psi& psi){return psi.alpha_array.to_pytensor_1d();},
            [](Psi& psi, const real_tensor<1u>& input) {psi.alpha_array = input;}
        )
        .def_property(
            "beta",
            [](const Psi& psi){return psi.beta_array.to_pytensor_1d();},
            [](Psi& psi, const real_tensor<1u>& input) {psi.beta_array = input;}
        )
        .def_property(
            "b",
            [](const Psi& psi){return psi.b_array.to_pytensor_1d();},
            [](Psi& psi, const complex_tensor<1u>& input) {psi.b_array = input; psi.update_kernel();}
        )
        .def_property(
            "W",
            [](const Psi& psi){return psi.W_array.to_pytensor_2d(shape_t<2u>{psi.N, psi.M});},
            [](Psi& psi, const complex_tensor<2u>& input) {psi.W_array = input; psi.update_kernel();}
        )
        .def_readonly("num_params", &Psi::num_params)
        .def_property("params", &Psi::get_params_py, &Psi::set_params_py)
        .def_property_readonly("free_quantum_axis", [](const Psi& psi) {return psi.free_quantum_axis;})
        .def_property_readonly("num_angles", &Psi::get_num_angles);

#endif // ENABLE_PSI

#ifdef ENABLE_PSI_DEEP
    py::class_<PsiDeep>(m, "PsiDeep")
        .def(py::init<
            const real_tensor<1u>&,
            const real_tensor<1u>&,
            const vector<complex_tensor<1u>>&,
            const vector<xt::pytensor<unsigned int, 2u>>&,
            const vector<complex_tensor<2u>>&,
            const complex_tensor<1u>&,
            const double,
            const bool,
            const bool
        >())
        .def("copy", &PsiDeep::copy)
        .def_readwrite("prefactor", &PsiDeep::prefactor)
        .def_readwrite("translational_invariance", &PsiDeep::translational_invariance)
        .def_readwrite("N_i", &PsiDeep::N_i)
        .def_readwrite("N_j", &PsiDeep::N_j)
        .def_readonly("gpu", &PsiDeep::gpu)
        .def_readonly("N", &PsiDeep::N)
        .def_readonly("num_params", &PsiDeep::num_params)
        .def_property(
            "params",
            [](const PsiDeep& psi) {return psi.get_params().to_pytensor_1d();},
            [](PsiDeep& psi, const complex_tensor<1u>& new_params) {psi.set_params(Array<complex_t>(new_params, false));}
        )
        .def_property_readonly("alpha", [](const PsiDeep& psi) {return psi.alpha_array.to_pytensor_1d();})
        .def_property_readonly("beta", [](const PsiDeep& psi) {return psi.beta_array.to_pytensor_1d();})
        .def_property_readonly("b", &PsiDeep::get_b)
        .def_property_readonly("connections", &PsiDeep::get_connections)
        .def_property_readonly("W", &PsiDeep::get_W)
        .def_property_readonly("final_weights", [](const PsiDeep& psi) {return psi.final_weights.to_pytensor_1d();})
    #ifdef ENABLE_EXACT_SUMMATION
        .def_property_readonly("_vector", [](const PsiDeep& psi) {return psi_vector_py(psi);})
        .def("norm", [](const PsiDeep& psi, ExactSummation& exact_summation) {return psi_norm(psi, exact_summation);})
    #endif // ENABLE_EXACT_SUMMATION
        .def_property_readonly("free_quantum_axis", [](const PsiDeep& psi) {return psi.free_quantum_axis;});

#endif // ENABLE_PSI_DEEP

#ifdef ENABLE_PSI_PAIR

    py::class_<PsiPair>(m, "PsiPair")
        .def(py::init<
            const real_tensor<1u>&,
            const real_tensor<1u>&,
            const vector<complex_tensor<1u>>&,
            const vector<xt::pytensor<unsigned int, 2u>>&,
            const vector<complex_tensor<2u>>&,
            const complex_tensor<1u>&,
            const double,
            const bool,
            const bool
        >())
        .def("copy", &PsiPair::copy)
        .def_readwrite("prefactor", &PsiPair::prefactor)
        .def_readonly("gpu", &PsiPair::gpu)
        .def_property_readonly("N", [](const PsiPair& psi){return psi.psi_real.N;})
        .def_property_readonly("num_params", [](const PsiPair& psi){return psi.psi_real.num_params;})
        .def_property(
            "params",
            [](const PsiPair& psi) {return psi.get_params().to_pytensor_1d();},
            [](PsiPair& psi, const complex_tensor<1u>& new_params) {psi.set_params(Array<complex_t>(new_params, false));}
        )
        .def_property_readonly("alpha", [](const PsiPair& psi) {return psi.alpha_array.to_pytensor_1d();})
        .def_property_readonly("beta", [](const PsiPair& psi) {return psi.beta_array.to_pytensor_1d();})
    #ifdef ENABLE_EXACT_SUMMATION
        .def_property_readonly("_vector", [](const PsiPair& psi) {return psi_vector_py(psi);})
        .def("norm", [](const PsiPair& psi, ExactSummation& exact_summation) {return psi_norm(psi, exact_summation);})
    #endif // ENABLE_EXACT_SUMMATION
        .def_property_readonly("free_quantum_axis", [](const PsiPair& psi) {return psi.free_quantum_axis;});

#endif // ENABLE_PSI_PAIR

#ifdef ENABLE_PSI_CLASSICAL
    py::class_<PsiClassical>(m, "PsiClassical")
         .def(py::init<
             const string,
             const int,
             const unsigned int,
             const bool
         >())
        .def_readonly("gpu", &PsiClassical::gpu)
        .def_readonly("N", &PsiClassical::N)
    #ifdef ENABLE_EXACT_SUMMATION
        .def_property_readonly("vector", [](const PsiClassical& psi) {return psi_vector(psi).to_pytensor_1d();})
    #endif // ENABLE_EXACT_SUMMATION
        .def_readonly("num_params", &PsiClassical::num_params)
        .def_property_readonly("free_quantum_axis", [](const PsiClassical& psi) {return psi.free_quantum_axis;})
        .def_property_readonly("num_angles", &PsiClassical::get_num_angles);

#endif // ENABLE_PSI_CLASSICAL

#ifdef ENABLE_PSI_EXACT
    py::class_<PsiExact>(m, "PsiExact")
        .def(py::init<
           const complex_tensor<1u>&,
           const unsigned int,
           const bool
        >())
        .def_readonly("gpu", &PsiExact::gpu)
        .def_readonly("N", &PsiExact::N)
    #ifdef ENABLE_EXACT_SUMMATION
        .def_property_readonly("vector", [](const PsiExact& psi) {return psi_vector(psi).to_pytensor_1d();})
        .def("norm", [](const PsiExact& psi, ExactSummation& exact_summation) {return psi_norm(psi, exact_summation);})
    #endif // ENABLE_EXACT_SUMMATION
        .def_readonly("num_params", &PsiExact::num_params)
        .def_readwrite("prefactor", &PsiExact::prefactor)
        .def_property_readonly("free_quantum_axis", [](const PsiExact& psi) {return psi.free_quantum_axis;})
        .def_property_readonly("num_angles", &PsiExact::get_num_angles);

#endif // ENABLE_PSI_EXACT

#ifdef ENABLE_PSI_DEEP_MIN
    py::class_<PsiDeepMin>(m, "PsiDeepMin")
        .def(py::init<
            const string
        >())
        .def_readonly("N", &PsiDeepMin::N)
    #ifdef ENABLE_EXACT_SUMMATION
        .def_property_readonly("vector", [](const PsiDeepMin& psi) {return psi_vector(psi).to_pytensor_1d();})
    #endif
        .def_property(
            "prefactor",
            [](const PsiDeepMin& psi) {return psi.prefactor;},
            [](PsiDeepMin& psi, const double new_value) {psi.prefactor = new_value; psi.log_prefactor = log(new_value);}
        )
        .def("log_psi_s", &PsiDeepMin::log_psi_s);

#endif // ENABLE_PSI_DEEP_MIN

#ifdef ENABLE_PSI_HAMILTONIAN
    py::class_<PsiHamiltonian>(m, "PsiHamiltonian")
        .def(py::init<
            const unsigned int,
            const Operator&
        >())
        .def_readonly("gpu", &PsiHamiltonian::gpu)
        .def_readonly("N", &PsiHamiltonian::N);

#endif // ENABLE_PSI_HAMILTONIAN

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
        .def(py::init<rbm_on_gpu::Spins::type, const unsigned int>())
        .def("array", &rbm_on_gpu::Spins::array)
        .def("flip", &rbm_on_gpu::Spins::flip)
        .def("rotate_left", &rbm_on_gpu::Spins::rotate_left)
        .def("shift_2d", &rbm_on_gpu::Spins::shift_2d);

#ifdef ENABLE_MONTE_CARLO
    py::class_<MonteCarloLoop>(m, "MonteCarloLoop")
        .def(py::init<unsigned int, unsigned int, unsigned int, unsigned int, bool>())
        .def(py::init<MonteCarloLoop&>())
        .def("set_total_z_symmetry", &MonteCarloLoop::set_total_z_symmetry)
        .def("set_fast_sweep", &MonteCarloLoop::set_fast_sweep)
        .def_property_readonly("num_steps", &MonteCarloLoop::get_num_steps)
        .def_property_readonly("acceptance_rate", [](const MonteCarloLoop& mc){
            return float(mc.acceptances_ar.front()) / float(mc.acceptances_ar.front() + mc.rejections_ar.front());
        });
#endif // ENABLE_MONTE_CARLO

#ifdef ENABLE_EXACT_SUMMATION
    py::class_<ExactSummation>(m, "ExactSummation")
        .def(py::init<unsigned int, bool>())
        .def("set_total_z_symmetry", &ExactSummation::set_total_z_symmetry)
        .def_property_readonly("num_steps", &ExactSummation::get_num_steps);
#endif // ENABLE_EXACT_SUMMATION


    py::class_<ExpectationValue>(m, "ExpectationValue")
        .def(py::init<bool>())
#ifdef ENABLE_MONTE_CARLO
#ifdef ENABLE_PSI
        .def("__call__", &ExpectationValue::__call__<Psi, MonteCarloLoop>)
        .def("__call__", &ExpectationValue::__call__vector<Psi, MonteCarloLoop>)
        .def("gradient", &ExpectationValue::gradient_py<Psi, MonteCarloLoop>)
        .def("fluctuation", &ExpectationValue::fluctuation<Psi, MonteCarloLoop>)
        .def("fluctuation_gradient", &ExpectationValue::fluctuation_gradient_py<Psi, MonteCarloLoop>)
        .def("difference", &ExpectationValue::difference<Psi, MonteCarloLoop>)
#endif // ENABLE_PSI
#ifdef ENABLE_PSI_DEEP
        .def("__call__", &ExpectationValue::__call__<PsiDeep, MonteCarloLoop>)
        .def("__call__", &ExpectationValue::__call__vector<PsiDeep, MonteCarloLoop>)
        .def("corrected", &ExpectationValue::corrected<PsiDeep, MonteCarloLoop>)
        .def("gradient", &ExpectationValue::gradient_py<PsiDeep, MonteCarloLoop>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiDeep, MonteCarloLoop>)
        .def("fluctuation_gradient", &ExpectationValue::fluctuation_gradient_py<PsiDeep, MonteCarloLoop>)
#endif // ENABLE_PSI_DEEP
#ifdef ENABLE_PSI_PAIR
        .def("__call__", &ExpectationValue::__call__<PsiPair, MonteCarloLoop>)
#endif // ENABLE_PSI_PAIR
#endif // ENABLE_MONTE_CARLO
#ifdef ENABLE_EXACT_SUMMATION
#ifdef ENABLE_PSI
        .def("__call__", &ExpectationValue::__call__<Psi, ExactSummation>)
        .def("__call__", &ExpectationValue::__call__vector<Psi, ExactSummation>)
        .def("gradient", &ExpectationValue::gradient_py<Psi, ExactSummation>)
        .def("fluctuation", &ExpectationValue::fluctuation<Psi, ExactSummation>)
        .def("fluctuation_gradient", &ExpectationValue::fluctuation_gradient_py<Psi, ExactSummation>)
        .def("difference", &ExpectationValue::difference<Psi, ExactSummation>)
#endif // ENABLE_PSI
#ifdef ENABLE_PSI_DEEP
        .def("__call__", &ExpectationValue::__call__<PsiDeep, ExactSummation>)
        .def("__call__", &ExpectationValue::__call__vector<PsiDeep, ExactSummation>)
        .def("corrected", &ExpectationValue::corrected<PsiDeep, ExactSummation>)
        .def("gradient", &ExpectationValue::gradient_py<PsiDeep, ExactSummation>)
        .def("fluctuation", &ExpectationValue::fluctuation<PsiDeep, ExactSummation>)
        .def("fluctuation_gradient", &ExpectationValue::fluctuation_gradient_py<PsiDeep, ExactSummation>)
#endif // ENABLE_PSI_DEEP
#ifdef ENABLE_PSI_PAIR
        .def("__call__", &ExpectationValue::__call__<PsiPair, ExactSummation>)
#endif // ENABLE_PSI_PAIR
#endif // ENABLE_EXACT_SUMMATION
    ;


    py::class_<HilbertSpaceDistance>(m, "HilbertSpaceDistance")
        .def(py::init<unsigned int, unsigned int, bool>())
#ifdef ENABLE_MONTE_CARLO
#ifdef ENABLE_PSI
        .def("__call__", &HilbertSpaceDistance::distance<Psi, Psi, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<Psi, Psi, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "nu"_a)
#endif // ENABLE_PSI
#ifdef ENABLE_PSI_DEEP
        .def("__call__", &HilbertSpaceDistance::distance<PsiDeep, PsiDeep, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<PsiDeep, PsiDeep, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "nu"_a)
        .def(
            "distance_2nd_order",
            [](HilbertSpaceDistance& hs_distance, const PsiDeep& psi, const PsiDeep& psi_prime, const quantum_expression::PauliExpression& expr, MonteCarloLoop& spin_ensemble) {
                return hs_distance.distance_2nd_order(
                    psi,
                    psi_prime,
                    Operator(expr, hs_distance.gpu),
                    Operator(expr * expr, hs_distance.gpu),
                    spin_ensemble
                );
            }
        )
#ifdef ENABLE_PSI_CLASSICAL
        .def("__call__", &HilbertSpaceDistance::distance<PsiClassical, PsiDeep, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<PsiClassical, PsiDeep, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "nu"_a)
#endif // ENABLE_PSI_CLASSICAL
#ifdef ENABLE_PSI_EXACT
        .def("__call__", &HilbertSpaceDistance::distance<PsiExact, PsiDeep, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<PsiExact, PsiDeep, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "nu"_a)
#endif // ENABLE_PSI_CLASSICAL
#endif // ENABLE_PSI_DEEP
#ifdef ENABLE_PSI_PAIR
        .def("__call__", &HilbertSpaceDistance::distance<PsiPair, PsiPair, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<PsiPair, PsiPair, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "nu"_a)
#ifdef ENABLE_PSI_CLASSICAL
        .def("gradient", &HilbertSpaceDistance::gradient_py<PsiClassical, PsiPair, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "nu"_a)
#endif // ENABLE_PSI_CLASSICAL
#endif // ENABLE_PSI_PAIR
#endif // ENABLE_MONTE_CARLO
#ifdef ENABLE_EXACT_SUMMATION
#ifdef ENABLE_PSI
        .def("__call__", &HilbertSpaceDistance::distance<Psi, Psi, ExactSummation>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<Psi, Psi, ExactSummation>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "nu"_a)
#endif // ENABLE_PSI
#ifdef ENABLE_PSI_DEEP
        .def("__call__", &HilbertSpaceDistance::distance<PsiDeep, PsiDeep, ExactSummation>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<PsiDeep, PsiDeep, ExactSummation>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "nu"_a)
        .def(
            "distance_2nd_order",
            [](HilbertSpaceDistance& hs_distance, const PsiDeep& psi, const PsiDeep& psi_prime, const quantum_expression::PauliExpression& expr, ExactSummation& spin_ensemble) {
                return hs_distance.distance_2nd_order(
                    psi,
                    psi_prime,
                    Operator(expr, hs_distance.gpu),
                    Operator(expr * expr, hs_distance.gpu),
                    spin_ensemble
                );
            }
        )
#ifdef ENABLE_PSI_CLASSICAL
        .def("__call__", &HilbertSpaceDistance::distance<PsiClassical, PsiDeep, ExactSummation>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<PsiClassical, PsiDeep, ExactSummation>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "nu"_a)
#endif // ENABLE_PSI_CLASSICAL
#ifdef ENABLE_PSI_EXACT
        .def("__call__", &HilbertSpaceDistance::distance<PsiExact, PsiDeep, ExactSummation>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<PsiExact, PsiDeep, ExactSummation>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "nu"_a)
#endif // ENABLE_PSI_CLASSICAL
#endif // ENABLE_PSI_DEEP
#ifdef ENABLE_PSI_PAIR
        .def("__call__", &HilbertSpaceDistance::distance<PsiPair, PsiPair, ExactSummation>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a)
        .def("gradient", &HilbertSpaceDistance::gradient_py<PsiPair, PsiPair, ExactSummation>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "nu"_a)
#ifdef ENABLE_PSI_CLASSICAL
        .def("gradient", &HilbertSpaceDistance::gradient_py<PsiClassical, PsiPair, ExactSummation>, "psi"_a, "psi_prime"_a, "operator_"_a, "is_unitary"_a, "spin_ensemble"_a, "nu"_a)
#endif // ENABLE_PSI_CLASSICAL
#endif // ENABLE_PSI_PAIR
#endif // ENABLE_EXACT_SUMMATION
    ;


    py::class_<KullbackLeibler>(m, "KullbackLeibler")
        .def(py::init<unsigned int, bool>())
#ifdef ENABLE_MONTE_CARLO
#ifdef ENABLE_PSI
        // .def("__call__", &KullbackLeibler::value<Psi, Psi, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "spin_ensemble"_a)
        // .def("gradient", &KullbackLeibler::gradient_py<Psi, Psi, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "spin_ensemble"_a)
#endif // ENABLE_PSI
#ifdef ENABLE_PSI_DEEP
        // .def("__call__", &KullbackLeibler::value<PsiDeep, PsiDeep, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "spin_ensemble"_a)
        // .def("gradient", &KullbackLeibler::gradient_py<PsiDeep, PsiDeep, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "spin_ensemble"_a)
#ifdef ENABLE_PSI_CLASSICAL
        .def("__call__", &KullbackLeibler::value<PsiClassical, PsiDeep, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "spin_ensemble"_a)
        .def("gradient", &KullbackLeibler::gradient_py<PsiClassical, PsiDeep, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "spin_ensemble"_a, "nu"_a)
#endif // ENABLE_PSI_CLASSICAL
#ifdef ENABLE_PSI_EXACT
        .def("__call__", &KullbackLeibler::value<PsiExact, PsiDeep, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "spin_ensemble"_a)
        .def("gradient", &KullbackLeibler::gradient_py<PsiExact, PsiDeep, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "spin_ensemble"_a, "nu"_a)
#endif // ENABLE_PSI_EXACT
#endif // ENABLE_PSI_DEEP
#ifdef ENABLE_PSI_PAIR
        // .def("__call__", &KullbackLeibler::value<PsiPair, PsiPair, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "spin_ensemble"_a)
        // .def("gradient", &KullbackLeibler::gradient_py<PsiPair, PsiPair, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "spin_ensemble"_a, "nu"_a)
#ifdef ENABLE_PSI_CLASSICAL
        .def("__call__", &KullbackLeibler::value<PsiClassical, PsiPair, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "spin_ensemble"_a)
        // .def("gradient", &KullbackLeibler::gradient_py<PsiClassical, PsiPair, MonteCarloLoop>, "psi"_a, "psi_prime"_a, "spin_ensemble"_a, "nu"_a)
#endif // ENABLE_PSI_CLASSICAL
#endif // ENABLE_PSI_PAIR
#endif // ENABLE_MONTE_CARLO
#ifdef ENABLE_EXACT_SUMMATION
#ifdef ENABLE_PSI
        // .def("__call__", &KullbackLeibler::value<Psi, Psi, ExactSummation>, "psi"_a, "psi_prime"_a, "spin_ensemble"_a)
        // .def("gradient", &KullbackLeibler::gradient_py<Psi, Psi, ExactSummation>, "psi"_a, "psi_prime"_a, "spin_ensemble"_a, "nu"_a)
#endif // ENABLE_PSI
#ifdef ENABLE_PSI_DEEP
        // .def("__call__", &KullbackLeibler::value<PsiDeep, PsiDeep, ExactSummation>, "psi"_a, "psi_prime"_a, "spin_ensemble"_a)
        // .def("gradient", &KullbackLeibler::gradient_py<PsiDeep, PsiDeep, ExactSummation>, "psi"_a, "psi_prime"_a, "spin_ensemble"_a, "nu"_a)
#ifdef ENABLE_PSI_CLASSICAL
        .def("__call__", &KullbackLeibler::value<PsiClassical, PsiDeep, ExactSummation>, "psi"_a, "psi_prime"_a, "spin_ensemble"_a)
        .def("gradient", &KullbackLeibler::gradient_py<PsiClassical, PsiDeep, ExactSummation>, "psi"_a, "psi_prime"_a, "spin_ensemble"_a, "nu"_a)
#endif // ENABLE_PSI_CLASSICAL
#ifdef ENABLE_PSI_EXACT
        .def("__call__", &KullbackLeibler::value<PsiExact, PsiDeep, ExactSummation>, "psi"_a, "psi_prime"_a, "spin_ensemble"_a)
        .def("gradient", &KullbackLeibler::gradient_py<PsiExact, PsiDeep, ExactSummation>, "psi"_a, "psi_prime"_a, "spin_ensemble"_a, "nu"_a)
#endif // ENABLE_PSI_EXACT
#endif // ENABLE_PSI_DEEP
#ifdef ENABLE_PSI_PAIR
        // .def("__call__", &KullbackLeibler::value<PsiPair, PsiPair, ExactSummation>, "psi"_a, "psi_prime"_a, "spin_ensemble"_a)
        // .def("gradient", &KullbackLeibler::gradient_py<PsiPair, PsiPair, ExactSummation>, "psi"_a, "psi_prime"_a, "spin_ensemble"_a, "nu"_a)
#ifdef ENABLE_PSI_CLASSICAL
        .def("__call__", &KullbackLeibler::value<PsiClassical, PsiPair, ExactSummation>, "psi"_a, "psi_prime"_a, "spin_ensemble"_a)
        // .def("gradient", &KullbackLeibler::gradient_py<PsiClassical, PsiPair, ExactSummation>, "psi"_a, "psi_prime"_a, "spin_ensemble"_a, "nu"_a)
#endif // ENABLE_PSI_CLASSICAL
#endif // ENABLE_PSI_PAIR
#endif // ENABLE_EXACT_SUMMATION
    ;

    py::class_<RenyiCorrelation>(m, "RenyiCorrelation")
        .def(py::init<bool>())
#ifdef ENABLE_SPECIAL_MONTE_CARLO
#ifdef ENABLE_PSI_DEEP
        .def("__call__", &RenyiCorrelation::__call__<PsiDeep, SpecialMonteCarloLoop>)
#endif // ENABLE_PSI_DEEP
#endif // ENABLE_SPECIAL_MONTE_CARLO
#ifdef ENABLE_SPECIAL_EXACT_SUMMATION
#ifdef ENABLE_PSI_DEEP
        .def("__call__", &RenyiCorrelation::__call__<PsiDeep, SpecialExactSummation>)
#endif // ENABLE_PSI_DEEP
#endif // ENABLE_SPECIAL_EXACT_SUMMATION
    ;


#ifdef ENABLE_PSI
    m.def("psi_O_k_vector", psi_O_k_vector_py<Psi>);
#endif

#ifdef ENABLE_PSI_DEEP
    m.def("psi_O_k_vector", psi_O_k_vector_py<PsiDeep>);
#endif

#ifdef ENABLE_PSI_PAIR
    // m.def("psi_O_k_vector", psi_O_k_vector_py<PsiPair>);
#endif

#if defined(ENABLE_PSI) && defined(ENABLE_EXACT_SUMMATION)
    m.def("get_S_matrix", [](const Psi& psi, ExactSummation& spin_ensemble){
        return get_S_matrix(psi, spin_ensemble).to_pytensor_2d(shape_t<2u>{psi.num_params, psi.num_params});
    });
#endif

    // m.def("get_O_k_vector", [](const Psi& psi, ExactSummation& spin_ensemble) {
    //     auto result_and_result_std = psi_O_k_vector(psi, spin_ensemble);
    //     return make_pair(
    //         result_and_result_std.first.to_pytensor_1d(),
    //         result_and_result_std.second.to_pytensor_1d()
    //     );
    // });
    // m.def("get_O_k_vector", [](const Psi& psi, MonteCarloLoop& spin_ensemble) {
    //     auto result_and_result_std = psi_O_k_vector(psi, spin_ensemble);
    //     return make_pair(
    //         result_and_result_std.first.to_pytensor_1d(),
    //         result_and_result_std.second.to_pytensor_1d()
    //     );
    // });
    // m.def("get_O_k_vector", [](const PsiDeep& psi, ExactSummation& spin_ensemble) {
    //     auto result_and_result_std = psi_O_k_vector(psi, spin_ensemble);
    //     return make_pair(
    //         result_and_result_std.first.to_pytensor_1d(),
    //         result_and_result_std.second.to_pytensor_1d()
    //     );
    // });
    // m.def("get_O_k_vector", [](const PsiDeep& psi, MonteCarloLoop& spin_ensemble) {
    //     auto result_and_result_std = psi_O_k_vector(psi, spin_ensemble);
    //     return make_pair(
    //         result_and_result_std.first.to_pytensor_1d(),
    //         result_and_result_std.second.to_pytensor_1d()
    //     );
    // });

    // m.def("psi_angles", [](const PsiDeep& psi, ExactSummation& spin_ensemble) {
    //     auto result_and_result_std = psi_angles(psi, spin_ensemble);
    //     return make_pair(
    //         result_and_result_std.first.to_pytensor_1d(),
    //         result_and_result_std.second.to_pytensor_1d()
    //     );
    // });
    // m.def("psi_angles", [](const PsiDeep& psi, MonteCarloLoop& spin_ensemble) {
    //     auto result_and_result_std = psi_angles(psi, spin_ensemble);
    //     return make_pair(
    //         result_and_result_std.first.to_pytensor_1d(),
    //         result_and_result_std.second.to_pytensor_1d()
    //     );
    // });

    m.def("activation_function", [](const complex<double>& x) {
        return my_logcosh(complex_t(x.real(), x.imag())).to_std();
    });

    m.def("setDevice", setDevice);
    m.def("start_profiling", start_profiling);
    m.def("stop_profiling", stop_profiling);
}
