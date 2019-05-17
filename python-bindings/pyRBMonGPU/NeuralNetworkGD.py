import numpy as np
from pathlib import Path
import pyRBMonGPU
from .NeuralNetwork import NeuralNetwork
import gradient_descent as gd
from ParallelAverage import SimpleFlock
from json_chain import EncoderChain
from json_numpy import NumpyEncoder
from .json_operator import OperatorEncoder
from itertools import islice
from scipy.ndimage import gaussian_filter1d
from PauliExpression import PauliExpression
import json


class DidNotConverge(RuntimeError):
    pass


class NeuralNetworkGD:
    def __init__(self, hilbert_space_distance, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)

        self.hilbert_space_distance = hilbert_space_distance

    def copy(self):
        result = super(self.__class__, self).copy()
        result.__class__ = self.__class__
        result.hilbert_space_distance = self.hilbert_space_distance
        return result

    def distance(self, step, operator, is_unitary):
        if not isinstance(operator, pyRBMonGPU.Operator):
            operator = CompiledOperator(operator, self.gpu)

        result = self.hilbert_space_distance(self, self, operator, is_unitary, self.spin_ensemble, False)
        if self.regularization is not None:
            result += self.regularization.cost(step, self)

        return result

    @property
    def smoothed_distance_history(self):
        distances = np.array(self.distance_history)
        sigma = 3 if self.mc else 1
        return gaussian_filter1d(distances, sigma, mode='nearest')

    @property
    def slopes(self):
        distances = self.smoothed_distance_history
        return abs(distances[1:] - distances[:-1])

    def push_params(self, params):
        stack_size = 3

        if not hasattr(self, "last_params_stack"):
            self.last_params_stack = np.empty((stack_size, self.num_active_params), dtype=complex)
            self.last_params_idx = 0

        self.last_params_stack[self.last_params_idx] = params
        self.last_params_idx = (self.last_params_idx + 1) % stack_size

    @property
    def last_params_avg(self):
        return np.mean(self.last_params_stack, axis=0)

    def set_params_and_return_gradient(self, step, params, psi0, operator, is_unitary):
        if self.mc:
            self.push_params(params)

        self.active_params = params
        gradient, distance = self.hilbert_space_distance.gradient(
            psi0, self, operator, is_unitary, self.spin_ensemble
        )
        if self.regularization is not None:
            gradient += self.regularization.gradient(step, self)
            complexity = self.regularization.cost(step, self)
            self.verbose_distance_history.append((+distance, +complexity))
            distance += complexity

        self.distance_history.append(distance)
        # self.active_params_history.append(+active_params)

        return gradient

    def get_gradient_descent_algorithms(self, psi0, operator, is_unitary, eta):
        gamma = 0.9
        epsilon = 0.1
        beta1 = 0.9
        beta2 = 0.99
        psi0_params = psi0.active_params

        return [
            {
                "name": "adamax",
                "iter": gd.adamax_generator(
                    psi0_params,
                    lambda step, params: self.set_params_and_return_gradient(
                        step, params, psi0, operator, is_unitary
                    ),
                    beta1,
                    beta2,
                    eta,
                    epsilon
                )
            },
            {
                "name": "rmsprop",
                "iter": gd.rmsprop_generator(
                    psi0_params,
                    lambda step, params: self.set_params_and_return_gradient(
                        step, params, psi0, operator, is_unitary
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
            #             step, params, psi0, operator, is_unitary
            #         ),
            #         eta,
            #         gamma
            #     )
            # }
        ]

    def save_gd_report(self, algorithm, psi0_params, operator, is_unitary, eta, commentary=""):
        database_path = Path("/home/burau/gd_reports.json")
        database_path.touch()

        with SimpleFlock(str(Path("/home/burau/gdlock"))):
            with open(database_path, 'r+') as f:
                if database_path.stat().st_size == 0:
                    reports = []
                else:
                    reports = json.load(f)

                print("n:", len(reports), ",", len(operator.coefficients))

                reports.append({
                    "algorithm": algorithm["name"],
                    "psi0_params": psi0_params,
                    "operator": operator,
                    "is_unitary": is_unitary,
                    "distance_history": self.distance_history,
                    "verbose_distance_history": self.verbose_distance_history,
                    # "params_history": self.params_history,
                    "smoothed_distance_history": self.smoothed_distance_history,
                    "eta": eta,
                    "commentary": commentary
                })

                f.seek(0)
                json.dump(reports, f, indent=2, cls=EncoderChain(NumpyEncoder(), OperatorEncoder()))
                f.truncate()

    def do_the_gradient_descent(self, operator, is_unitary, eta):
        max_steps = 150
        relative_slope_threshold = lambda step: 1 / (1000 + 100 * 0.95**step)
        distance_threshold = 1e-4

        self.psi0 = self.copy()
        final_distances = []
        final_params = []

        print_result = False
        for algorithm in self.get_gradient_descent_algorithms(self.psi0, operator, is_unitary, eta):
            self.distance_history = []
            self.verbose_distance_history = []
            # self.params_history = []
            num_steps = 50 if self.mc else 15
            list(islice(algorithm["iter"], num_steps))

            smoothed_distance_history = self.smoothed_distance_history
            if smoothed_distance_history[-1] > smoothed_distance_history[0]:
                self.save_gd_report(algorithm, self.psi0.active_params, operator, is_unitary, eta)
                print(f"[{algorithm['name']}] unstable gradient descent")
                continue

            initial_distance = smoothed_distance_history[0]
            initial_slope = self.slopes[0]
            final_distance = smoothed_distance_history[-1]

            while (
                self.slopes[-1] / initial_slope > relative_slope_threshold(num_steps) and
                # final_distance > distance_threshold and
                num_steps < max_steps
            ):
                list(islice(algorithm["iter"], 5))
                num_steps += 5

            if (initial_distance > 1e-3 and final_distance > initial_distance):
                self.save_gd_report(algorithm, self.psi0.active_params, operator, is_unitary, eta)
                print(f"[{algorithm['name']}] ineffective gradient descent")
                print(f"initial_distance: {initial_distance}, final_distance: {final_distance}")
                continue

            final_distances.append(final_distance)
            final_params.append(self.last_params_avg if self.mc else self.active_params)

            if print_result:
                print(algorithm["name"], final_distance)

            self.save_gd_report(algorithm, self.psi0.active_params, operator, is_unitary, eta)
            if final_distance < initial_distance:
                best_algorithm_idx = min(range(len(final_params)), key=lambda i: final_distances[i])
                self.active_params = final_params[best_algorithm_idx]
                return

            # self.save_gd_report(algorithm, self.psi0.active_params, operator, is_unitary, eta)

            print(f"[{algorithm['name']}] gradient descent did not converge in {max_steps} steps")
            print(algorithm["name"], final_distance)
            print_result = True

        print("All gradient descent algorithms failed")
        self.active_params = self.psi0.active_params
        raise DidNotConverge()

    def optimize_for(self, operator, is_unitary=False, regularization=None, eta=None):
        norm_threshold = 1e-1

        if eta is None:
            eta = 1e-3 if self.mc else 5e-3

        # self.operator = operator
        self.regularization = regularization
        compiled_op = CompiledOperator(operator, self.gpu)
        # self.spin_ensemble = pyRBMonGPU.MonteCarloLoop(self.spin_ensemble)

        if is_unitary:
            success = False
            for tries in range(3):
                try:
                    self.do_the_gradient_descent(compiled_op, True, eta)
                    success = True
                    break
                except DidNotConverge:
                    eta /= 5
            assert success, "Could not achieve convergence."

        elif operator.max_norm > norm_threshold:
            print("divide [max_norm]")
            num_steps = int(operator.max_norm / norm_threshold) + 1
            fitted_operator = 1 / num_steps * operator
            for n in range(num_steps):
                self.optimize_for(fitted_operator, regularization=regularization)
        else:
            self.do_the_gradient_descent(compiled_op, False, eta)

        if not self.mc:
            self.prefactor /= self.norm

    def intelligent_optimize_for(self, operator, exp=True, regularization=None, reversed_order=False):
        self.regularization = regularization
        length_limit = 1
        # threshold = 1e-2

        terms = sorted(list(operator), key=lambda t: -t.max_norm)

        if exp:
            unitary_ops = []
            unitary_op = 1
            for term in terms:
                unitary_op *= term.exp(0)
                if isinstance(unitary_op, PauliExpression) and len(unitary_op) > length_limit:
                    unitary_ops.append(unitary_op)
                    unitary_op = 1
            if isinstance(unitary_op, PauliExpression) and not unitary_op.is_numeric:
                unitary_ops.append(unitary_op)

            for unitary_op in (reversed(unitary_ops) if reversed_order else unitary_ops):
                self.optimize_for(unitary_op, True, regularization)
        else:
            group_size = 6
            for i in range(0, len(terms), group_size):
                group = sum(terms[i: i + group_size])
                self.optimize_for(group, False, regularization)
