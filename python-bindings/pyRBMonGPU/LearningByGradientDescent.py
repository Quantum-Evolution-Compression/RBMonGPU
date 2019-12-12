import numpy as np
import math
import pyRBMonGPU
from pyRBMonGPU import Operator
import gradient_descent as gd
from itertools import islice
from scipy.ndimage import gaussian_filter1d
from QuantumExpression import PauliExpression
from HilbertSpaceCorrelations import HilbertSpaceCorrelations
from collections import namedtuple
from pathlib import Path
import json
import os


# list of actions
modes = namedtuple(
    "modes",
    "unitary_evolution groundstate"
)._make(range(2))


class DidNotConverge(RuntimeError):
    pass


class UnstableNetwork(RuntimeError):
    pass


class PoorLearning(RuntimeError):
    pass


class LearningByGradientDescent:
    def __init__(self, psi, spin_ensemble):
        self.gpu = psi.gpu
        self.mc = type(spin_ensemble) is pyRBMonGPU.MonteCarloLoop
        self.spin_ensemble = spin_ensemble
        self.num_params = psi.num_params
        self.hilbert_space_distance = pyRBMonGPU.HilbertSpaceDistance(self.num_params, psi.gpu)
        self.expectation_value = pyRBMonGPU.ExpectationValue(self.gpu)
        self.regularization = None
        self.eta = None

    @property
    def smoothed_distance_history(self):
        distances = np.array(self.distance_history)
        sigma = 3 if self.mc else 1
        return gaussian_filter1d(distances, sigma, mode='nearest')

    def smoothed(self, x):
        return gaussian_filter1d(np.array(x[-100:]), 5, mode='nearest')

    @property
    def slopes(self):
        distances = self.smoothed_distance_history
        return abs(distances[1:] - distances[:-1])

    @property
    def graph(self):
        return np.log(self.smoothed_distance_history)

    @property
    def is_highly_fluctuating(self):
        graph = np.log(self.distance_history[-50:])
        diff_graph = graph[1:] - graph[:-1]
        return np.std(diff_graph) > 0.25

    @property
    def is_descending(self):
        graph = self.graph[-50:]
        return graph[-1] - graph[0] < -0.05

    @property
    def is_learning_poorly(self):
        graph = self.graph
        return np.count_nonzero(graph > graph[0] + 0.2) > 40

    def has_converged(self, x, threshold=None):
        threshold = threshold or (3e-3 if self.mc else 1e-3)

        smoothed_x = self.smoothed(x)
        return abs(smoothed_x[0] - smoothed_x[-1]) < threshold

    def push_params(self, params):
        stack_size = 3

        if not hasattr(self, "last_params_stack"):
            self.last_params_stack = np.empty((stack_size, self.psi.num_params), dtype=complex)
            self.last_params_idx = 0

        self.last_params_stack[self.last_params_idx] = params
        self.last_params_idx = (self.last_params_idx + 1) % stack_size

    @property
    def last_params_avg(self):
        return np.mean(self.last_params_stack, axis=0)

    def set_params_and_return_gradient(self, step, params):
        # if self.mc:
        #     self.push_params(params)

        self.psi.params = params
        if not self.mc:
            self.psi.normalize(self.spin_ensemble)

        if self.mode == modes.unitary_evolution:
            gradient, distance = self.hilbert_space_distance.gradient(
                self.psi0, self.psi, self.operator, self.is_unitary, self.spin_ensemble
            )
            distance -= self.distance_0
            if distance < 0:
                gradient *= -1

            gradient *= self.gradient_prefactor
            if self.regularization is not None:
                gradient += self.regularization.gradient(step, self.psi)
                complexity = self.regularization.cost(step, self.psi)
                self.verbose_distance_history.append((+distance, +complexity))
                distance += complexity

            self.distance_history.append(distance)
        elif self.mode == modes.groundstate:
            if self.imaginary_time_evolution:
                gradient, distance = self.hilbert_space_distance.gradient(
                    self.psi, self.psi, self.opt_operator, True, self.spin_ensemble
                )
            else:
                gradient, energy = self.expectation_value.gradient(self.psi, self.operator, self.spin_ensemble)

            delta_energy, energy = self.expectation_value.fluctuation(self.psi, self.operator, self.spin_ensemble)
            self.delta_energy_history.append(delta_energy)

            energy = energy.real
            self.energy_history.append(energy)

            distances = []
            for excluded_state in self.excluded_states:
                es_gradient, distance = self.hilbert_space_distance.gradient(
                    excluded_state, self.psi, self.identity, True, self.spin_ensemble
                )
                distances.append(distance)

                gradient -= self.level_repulsion * max(1, min(10, math.log(distance.real))) * es_gradient

            self.excluded_states_distances.append(distances)

        if hasattr(self, "avoid_correlations"):
            if step < 100:
                gradient -= self.avoid_correlations * HilbertSpaceCorrelations(self.psi).gradient

        return gradient

    def get_gradient_descent_algorithm(self, name):
        psi_init_params = self.psi_init.params

        kwargs = {}
        if self.eta is not None:
            kwargs["eta"] = self.eta

        return {
            "padam": lambda: gd.padam_generator(
                psi_init_params,
                lambda step, params: self.set_params_and_return_gradient(
                    step, params
                ),
                **kwargs
            ),
            "nag": lambda: gd.nag_generator(
                psi_init_params,
                lambda step, params: self.set_params_and_return_gradient(
                    step, params
                ),
                **kwargs
            )
        }[name]()

    @property
    def min_report(self):
        result = {
            "operator": self.operator.expr.to_json()
        }
        if hasattr(self, "is_unitary"):
            result["is_unitary"] = self.is_unitary
        if hasattr(self, "distance_history"):
            result["distance_history"] = self.distance_history
        if hasattr(self, "unsuccessful_curves"):
            result["unsuccessful_curves"] = self.unsuccessful_curves
        if hasattr(self, "energy_history"):
            result["energy_history"] = self.energy_history
        if hasattr(self, "delta_energy_history"):
            result["delta_energy_history"] = self.delta_energy_history
        if hasattr(self, "excluded_states_distances"):
            result["excluded_states_distances"] = self.excluded_states_distances

        return result

    @property
    def report(self):
        result = self.min_report
        if hasattr(self, "psi0"):
            result["psi0"] = self.psi0.to_json()
        if hasattr(self, "psi"):
            result["psi"] = self.psi.to_json()

        return result

    def save_report(self):
        job_name = os.environ.get("JOB_NAME", "default")
        run_id = int(os.environ.get("RUN_ID", 0))
        folder = Path("/data3/burau/data") / job_name
        folder.mkdir(exist_ok=True)
        with open(folder / f"data_{run_id}.json", 'w+') as f:
            json.dump(self.report, f, indent=2)

    def find_low_laying_state(
        self, psi_init, operator, excluded_states=[], avoid_correlations=0,
        level_repulsion=1, imaginary_time_evolution=True, min_steps=200, max_steps=1500
    ):
        self.psi0 = psi_init
        self.psi = +psi_init
        self.identity = Operator(PauliExpression(0, 0), self.gpu)
        self.operator = Operator(operator, self.gpu)
        self.opt_operator = Operator(1 - 0.1 * operator, self.gpu)
        self.excluded_states = excluded_states
        self.energy_history = []
        self.delta_energy_history = []
        self.mode = modes.groundstate
        self.imaginary_time_evolution = imaginary_time_evolution
        self.excluded_states_distances = []
        self.avoid_correlations = avoid_correlations
        self.level_repulsion = level_repulsion

        algorithm = next(iter(self.get_gradient_descent_algorithm()))
        algorithm_iter = algorithm["iter"]()

        steps = min_steps
        list(islice(algorithm_iter, min_steps))

        while steps < max_steps and not self.has_converged(self.energy_history):
            list(islice(algorithm_iter, 100))
            steps += 100

        return self.psi

    def do_the_gradient_descent(self, algorithm_name="padam"):
        # max_steps = 400

        self.psi0 = self.psi.copy()
        algorithm = self.get_gradient_descent_algorithm(algorithm_name)
        self.distance_history = []
        self.verbose_distance_history = []
        self.gradient_prefactor = 1

        num_steps = 400
        list(islice(algorithm, num_steps))

        while num_steps <= 1000 and self.is_descending:
            list(islice(algorithm, 200))
            num_steps += 200

        # if self.is_learning_poorly:
        #     raise PoorLearning()

        # smoothed_distance_history = self.smoothed_distance_history
        # if smoothed_distance_history[0] > 1e-3 and smoothed_distance_history[-1] / smoothed_distance_history[0] > 1 / 3:
        #     raise PoorLearning()

        print(self.smoothed_distance_history[-1])

    def do_preconditioning(self, psi, psi_ref):
        print("preconditioning")
        id_op = Operator(PauliExpression(0, 0), self.gpu)

        def gradient(step, params):
            psi.params = params
            if not self.mc:
                psi.normalize(self.spin_ensemble)

            gradient, distance = self.hilbert_space_distance.gradient(
                psi_ref, psi, id_op, True, self.spin_ensemble
            )

            return gradient

        psi = +psi
        gd.padam(psi.params, 75, gradient)
        return psi

    def optimize_for(self, psi, psi_init, operator, is_unitary=False, regularization=None, psi_init_getter=None, eta=None, distance_0=0):
        self.psi = +psi
        self.psi_init = psi_init

        self.eta = eta
        self.distance_0 = distance_0
        self.regularization = regularization
        self.operator = Operator(operator, self.gpu)
        self.is_unitary = is_unitary
        self.mode = modes.unitary_evolution
        self.unsuccessful_curves = []

        id_op = Operator(PauliExpression(0, 0), self.gpu)

        if is_unitary:
            success = False
            for tries in range(2):
                try:
                    if (
                        self.psi_init is not psi and
                        self.hilbert_space_distance(self.psi, self.psi_init, id_op, True, self.spin_ensemble) > 0.05
                    ):
                        self.psi_init = self.do_preconditioning(self.psi_init, self.psi)

                    if self.psi_init is not psi:
                        self.do_the_gradient_descent("padam")
                    else:
                        self.do_the_gradient_descent("nag")

                    success = True
                    break
                except PoorLearning:
                    self.unsuccessful_curves.append(self.distance_history)
                    self.psi_init = psi_init_getter()

            if not success:
                self.save_report()
                raise DidNotConverge()

        # elif operator.max_norm > norm_threshold:
        #     print("divide [max_norm]")
        #     num_steps = int(operator.max_norm / norm_threshold) + 1
        #     fitted_operator = 1 / num_steps * operator
        #     for n in range(num_steps):
        #         self.optimize_for(fitted_operator, is_unitary=False, regularization=regularization, eta=eta)
        else:
            self.do_the_gradient_descent("padam")

        if not self.mc:
            self.psi.normalize(self.spin_ensemble)

        return self.psi

    def intelligent_optimize_for(self, psi, operator, exp=True, regularization=None, reversed_order=False):
        self.regularization = regularization
        length_limit = 1

        terms = sorted(list(operator), key=lambda t: -t.max_norm)

        if exp:
            unitary_ops = []
            unitary_op = 1
            for term in terms:
                unitary_op *= term.exp(0)
                # unitary_op = unitary_op.apply_threshold(1e-2)
                if isinstance(unitary_op, PauliExpression) and len(unitary_op) > length_limit:
                    unitary_ops.append(unitary_op)
                    unitary_op = 1
            if isinstance(unitary_op, PauliExpression) and not unitary_op.is_numeric:
                unitary_ops.append(unitary_op)

            for unitary_op in (reversed(unitary_ops) if reversed_order else unitary_ops):
                self.optimize_for(psi, unitary_op, True, regularization)
        else:
            group_size = 1
            for i in range(0, len(terms), group_size):
                group = sum(terms[i: i + group_size])
                self.optimize_for(psi, group, False, regularization)
