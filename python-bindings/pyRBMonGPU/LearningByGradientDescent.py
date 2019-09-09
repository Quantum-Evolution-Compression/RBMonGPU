import numpy as np
import pyRBMonGPU
from pyRBMonGPU import Operator
import gradient_descent as gd
from itertools import islice
from scipy.ndimage import gaussian_filter1d
from QuantumExpression import PauliExpression
from HilbertSpaceCorrelations import HilbertSpaceCorrelations
from collections import namedtuple


# list of actions
modes = namedtuple(
    "modes",
    "unitary_evolution groundstate"
)._make(range(2))


class DidNotConverge(RuntimeError):
    pass


class LearningByGradientDescent:
    def __init__(self, psi, spin_ensemble):
        self.gpu = psi.gpu
        self.mc = type(spin_ensemble) is pyRBMonGPU.MonteCarloLoop
        self.spin_ensemble = spin_ensemble
        self.num_params = psi.num_active_params
        self.hilbert_space_distance = pyRBMonGPU.HilbertSpaceDistance(self.num_params, psi.gpu)
        self.expectation_value = pyRBMonGPU.ExpectationValue(self.gpu)
        self.regularization = None

    @property
    def smoothed_distance_history(self):
        distances = np.array(self.distance_history)
        sigma = 3 if self.mc else 1
        return gaussian_filter1d(distances, sigma, mode='nearest')

    @property
    def slopes(self):
        distances = self.smoothed_distance_history
        return abs(distances[1:] - distances[:-1])

    @property
    def graph(self):
        return np.log(self.smoothed_distance_history[-50:])

    @property
    def is_highly_fluctuating(self):
        graph = self.graph
        diff_graph = graph[1:] - graph[:-1]
        return np.std(diff_graph) > 0.25

    @property
    def is_descending(self):
        graph = self.graph
        return graph[-1] - graph[0] < -0.05

    def push_params(self, params):
        stack_size = 3

        if not hasattr(self, "last_params_stack"):
            self.last_params_stack = np.empty((stack_size, self.psi.num_active_params), dtype=complex)
            self.last_params_idx = 0

        self.last_params_stack[self.last_params_idx] = params
        self.last_params_idx = (self.last_params_idx + 1) % stack_size

    @property
    def last_params_avg(self):
        return np.mean(self.last_params_stack, axis=0)

    def set_params_and_return_gradient(self, step, params):
        # if self.mc:
        #     self.push_params(params)

        self.psi.active_params = params
        if not self.mc:
            self.psi.normalize()

        if self.mode == modes.unitary_evolution:
            gradient, distance = self.hilbert_space_distance.gradient(
                self.psi0, self.psi, self.operator, self.is_unitary, self.spin_ensemble
            )

            gradient *= self.gradient_prefactor
            if self.regularization is not None:
                gradient += self.regularization.gradient(step, self.psi)
                complexity = self.regularization.cost(step, self.psi)
                self.verbose_distance_history.append((+distance, +complexity))
                distance += complexity

            if step < 100:
                gradient -= HilbertSpaceCorrelations(self.psi).gradient

            self.distance_history.append(distance)
        elif self.mode == modes.groundstate:
            gradient, energy = self.expectation_value.gradient(self.psi, self.operator, self.spin_ensemble)
            energy = energy.real
            if step > 100 and energy < self.min_energy:
                self.min_energy = energy
                self.min_params = +params

            self.energy_history.append(energy)

            for excluded_state in self.excludes_states:
                gradient -= self.hilbert_space_distance.gradient(
                    excluded_state, self.psi, self.identity, True, self.spin_ensemble
                )[0]

            if step < 250:
                gradient -= HilbertSpaceCorrelations(self.psi).gradient

        return gradient

    def get_gradient_descent_algorithms(self, eta):
        gamma = 0.9
        epsilon = 0.1
        beta1 = 0.9
        beta2 = 0.99
        psi0_params = self.psi0.active_params

        return [
            {
                "name": "adamax",
                "iter": lambda: gd.adamax_generator(
                    psi0_params,
                    lambda step, params: self.set_params_and_return_gradient(
                        step, params
                    ),
                    beta1,
                    beta2,
                    eta,
                    epsilon
                )
            },
            {
                "name": "rmsprop",
                "iter": lambda: gd.rmsprop_generator(
                    psi0_params,
                    lambda step, params: self.set_params_and_return_gradient(
                        step, params
                    ),
                    eta,
                    gamma,
                    epsilon
                )
            },
            # {
            #     "name": "nag",
            #     "iter": gd.nag_generator(
            #         psi0_params,
            #         lambda step, params: self.set_params_and_return_gradient(
            #             step, params
            #         ),
            #         eta,
            #         gamma
            #     )
            # }
        ]

    @property
    def min_report(self):
        result = {
            "operator": self.operator.expr.to_json()
        }
        if hasattr(self, "is_unitary"):
            result["is_unitary"] = self.is_unitary
        if hasattr(self, "distance_history"):
            result["distance_history"] = self.distance_history
        if hasattr(self, "energy_history"):
            result["energy_history"] = self.energy_history

        return result

    @property
    def report(self):
        result = self.min_report
        if hasattr(self, "psi0"):
            result["psi0"] = self.psi0.to_json()
        if hasattr(self, "psi"):
            result["psi"] = self.psi.to_json()

        return result

    def find_low_laying_state(self, psi_init, operator, excludes_states=[], eta=1e-3):
        self.psi0 = psi_init
        self.psi = +psi_init
        self.identity = Operator(PauliExpression(0, 0), self.gpu)
        self.operator = Operator(operator, self.gpu)
        self.excludes_states = excludes_states
        self.energy_history = []
        self.min_energy = float("+inf")
        self.mode = modes.groundstate

        algorithm = next(iter(self.get_gradient_descent_algorithms(eta)))
        algorithm_iter = algorithm["iter"]()

        list(islice(algorithm_iter, 500))

        self.psi.active_params = self.min_params
        if not self.mc:
            self.psi.normalize()

        return self.psi

    def do_the_gradient_descent(self, eta):
        # max_steps = 400

        self.psi0 = self.psi.copy()
        for algorithm in self.get_gradient_descent_algorithms(eta):
            self.sample_psi_prime = False
            self.distance_history = []
            self.verbose_distance_history = []
            self.gradient_prefactor = 1
            # self.params_history = []

            num_steps = 200
            algorithm_iter = algorithm["iter"]()
            list(islice(algorithm_iter, num_steps))

            smoothed_distance_history = self.smoothed_distance_history
            initial_distance = smoothed_distance_history[0]

            while num_steps <= 1000 and self.is_descending:
                list(islice(algorithm_iter, 200))
                num_steps += 200

            # v2
            if (
                initial_distance > 2e-4 and self.smoothed_distance_history[-1] / initial_distance > 0.1
            ):
                params_at_mark_A = self.psi.active_params
                self.gradient_prefactor = 1 / 2 if self.is_highly_fluctuating else 2
                list(islice(algorithm_iter, 300))
                num_steps += 300
                while num_steps <= 1200 and self.is_descending:
                    list(islice(algorithm_iter, 200))
                    num_steps += 200

                if self.smoothed_distance_history[-1] / initial_distance > 0.1:
                    self.psi.active_params = params_at_mark_A

            # if self.smoothed_distance_history[-1] >= initial_distance:
            #     self.save_gd_report(algorithm, self.psi0.active_params, eta)

            print(self.smoothed_distance_history[-1] / initial_distance)

            return
            # final_distance = self.smoothed_distance_history[-1]

            # final_distances.append(final_distance)
            # final_params.append(self.last_params_avg if self.mc else self.psi.active_params)

            # if print_result:
            #     print(algorithm["name"], final_distance)

            # if (
            #     (initial_distance > 2e-4 and final_distance / initial_distance < 0.1) or
            #     (initial_distance <= 2e-4 and final_distance < initial_distance)
            # ):
            #     best_algorithm_idx = min(range(len(final_params)), key=lambda i: final_distances[i])
            #     self.psi.active_params = final_params[best_algorithm_idx]
            #     return

            # self.save_gd_report(algorithm, self.psi0.active_params, eta)

            # print(f"[{algorithm['name']}] gradient descent did not converge in {num_steps} steps")
            # print(algorithm["name"], final_distance)
            # print_result = True

        print("All gradient descent algorithms failed")
        self.psi.active_params = self.psi0.active_params
        raise DidNotConverge()

    def optimize_for(self, psi, operator, is_unitary=False, regularization=None, eta=None):
        self.psi = psi
        norm_threshold = 1e-1

        eta = eta or 1e-3

        self.regularization = regularization
        self.operator = Operator(operator, self.gpu)
        self.is_unitary = is_unitary
        self.mode = modes.unitary_evolution

        if is_unitary:
            success = False
            for tries in range(3):
                try:
                    self.do_the_gradient_descent(eta)
                    success = True
                    break
                except DidNotConverge:
                    eta /= 5
            assert success, "Could not achieve convergence."

        # elif operator.max_norm > norm_threshold:
        #     print("divide [max_norm]")
        #     num_steps = int(operator.max_norm / norm_threshold) + 1
        #     fitted_operator = 1 / num_steps * operator
        #     for n in range(num_steps):
        #         self.optimize_for(fitted_operator, is_unitary=False, regularization=regularization, eta=eta)
        else:
            self.do_the_gradient_descent(eta)

        if not self.mc:
            self.psi.normalize()

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
