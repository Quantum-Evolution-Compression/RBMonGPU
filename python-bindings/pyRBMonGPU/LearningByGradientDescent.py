import numpy as np
from pathlib import Path
import pyRBMonGPU
from pyRBMonGPU import Operator
import gradient_descent as gd
from ParallelAverage import SimpleFlock
from json_chain import EncoderChain
from json_numpy import NumpyEncoder
from .json_operator import OperatorEncoder
from itertools import islice
from scipy.ndimage import gaussian_filter1d
from QuantumExpression import PauliExpression
import json


class DidNotConverge(RuntimeError):
    pass


class LearningByGradientDescent:
    def __init__(self, psi, spin_ensemble):
        self.psi = psi
        self.gpu = psi.gpu
        self.mc = type(spin_ensemble) is pyRBMonGPU.MonteCarloLoop
        self.spin_ensemble = spin_ensemble
        self.num_params = psi.num_active_params
        self.hilbert_space_distance = pyRBMonGPU.HilbertSpaceDistance(self.num_params, psi.gpu)
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
    def is_highly_fluctuating(self):
        graph = np.log(self.smoothed_distance_history[-50:])
        return np.std(graph) > 0.25

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
        if self.mc:
            self.push_params(params)

        self.psi.active_params = params
        gradient, distance = self.hilbert_space_distance.gradient(
            self.psi0, self.psi, self.operator, self.is_unitary, self.spin_ensemble
        )
        gradient[self.psi.active_params_types == 1] *= 10
        gradient *= self.gradient_prefactor
        if self.regularization is not None:
            gradient += self.regularization.gradient(step, self.psi)
            complexity = self.regularization.cost(step, self.psi)
            self.verbose_distance_history.append((+distance, +complexity))
            distance += complexity

        self.distance_history.append(distance)

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

    def save_gd_report(self, algorithm, psi0_params, eta, commentary=""):
        database_path = Path("/home/burau/gd_reports.json")
        database_path.touch()

        with SimpleFlock(str(Path("/home/burau/gdlock"))):
            with open(database_path, 'r+') as f:
                if database_path.stat().st_size == 0:
                    reports = []
                else:
                    reports = json.load(f)

                print("n:", len(reports), ",", len(self.operator.coefficients))

                reports.append({
                    "algorithm": algorithm["name"],
                    "psi0_params": psi0_params,
                    "operator": self.operator,
                    "is_unitary": self.is_unitary,
                    "distance_history": self.distance_history,
                    "verbose_distance_history": self.verbose_distance_history,
                    # "params_history": self.params_history,
                    "smoothed_distance_history": self.smoothed_distance_history,
                    "sample_psi_prime": self.sample_psi_prime,
                    "eta": eta,
                    "commentary": commentary
                })

                f.seek(0)
                json.dump(reports, f, indent=2, cls=EncoderChain(NumpyEncoder(), OperatorEncoder()))
                f.truncate()

    @property
    def min_report(self):
        result = {
            "operator": self.operator,
            "is_unitary": self.is_unitary,
            "distance_history": self.distance_history
        }

        return json.loads(json.dumps(
            result, cls=EncoderChain(NumpyEncoder(), OperatorEncoder())
        ))

    def do_the_gradient_descent(self, eta):
        # max_steps = 400

        self.psi0 = self.psi.copy()
        for algorithm in self.get_gradient_descent_algorithms(eta):
            self.sample_psi_prime = False
            self.distance_history = []
            self.verbose_distance_history = []
            self.gradient_prefactor = 1
            # self.params_history = []

            num_steps = 150
            algorithm_iter = algorithm["iter"]()
            list(islice(algorithm_iter, num_steps))

            smoothed_distance_history = self.smoothed_distance_history

            initial_distance = smoothed_distance_history[0]

            if (
                initial_distance > 2e-4 and self.smoothed_distance_history[-1] / initial_distance > 0.1
            ):
                params_at_mark_A = self.psi.active_params
                is_highly_fluctuating_at_mark_A = self.is_highly_fluctuating
                self.gradient_prefactor = 1 / 8 if is_highly_fluctuating_at_mark_A else 8
                list(islice(algorithm_iter, 200))
                num_steps += 200

                if self.smoothed_distance_history[-1] / initial_distance > 0.1:
                    self.psi.active_params = params_at_mark_A
                    self.gradient_prefactor = 1 / 4 if is_highly_fluctuating_at_mark_A else 4
                    list(islice(algorithm_iter, 400))
                    num_steps += 400
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

    def optimize_for(self, operator, is_unitary=False, regularization=None, eta=None):
        norm_threshold = 1e-1

        eta = eta or 1e-3

        self.regularization = regularization
        self.operator = Operator(operator, self.gpu)
        self.is_unitary = is_unitary

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
            self.psi.prefactor /= self.psi.norm

    def intelligent_optimize_for(self, operator, exp=True, regularization=None, reversed_order=False):
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
                self.optimize_for(unitary_op, True, regularization)
        else:
            group_size = 1
            for i in range(0, len(terms), group_size):
                group = sum(terms[i: i + group_size])
                self.optimize_for(group, False, regularization)
