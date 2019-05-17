from QuantumExpression import PauliExpression as pe
from pyRBMonGPU import ExpectationValue
import numpy as np
import random


class SpinConstraint:
    def __init__(self, lambda_, operator, spin_ensemble, gpu):
        operator = [term for term in operator if not term.is_numeric]

        self.lambda_ = lambda_
        self.expectation_value = ExpectationValue(gpu)
        self.spin_ensemble = spin_ensemble
        self.gpu = gpu

        self.constraints = []
        self.compiled_constraints = []
        self.create_single_spin_constraints(operator)

        long_terms = [term for term in operator if len(term.pauli_string) > 2]
        if long_terms:
            self.create_multiple_spin_constraints(long_terms, 10 * len(long_terms), 3)

    def gradient(self, step, psi):
        if step == 0:
            self.psi0 = psi.copy()

        diff = np.array(self.expectation_value.difference(self.psi0, psi, self.compiled_constraints, self.spin_ensemble))
        diff_sign = np.sign(diff)

        diff_operator = sum(2 * float(d.real) * constraint for d, constraint in zip(diff, self.constraints))

        return self.lambda_ * self.expectation_value.gradient(
            psi, CompiledOperator(diff_operator, psi.gpu), self.spin_ensemble
        )

    def cost(self, step, psi):
        if step == 0:
            self.psi0 = psi.copy()

        diff = np.array(self.expectation_value.difference(self.psi0, psi, self.compiled_constraints, self.spin_ensemble))

        return self.lambda_ * np.sum(abs(diff))

    def create_single_spin_constraints(self, operator):
        constraints = {}

        for term in operator:
            for i, sigma in term.pauli_string:
                if i not in constraints:
                    constraints[i] = sigma
                    continue

                if constraints[i] != sigma:
                    constraints[i] = -1

        constraints = [pe({i: sigma}) for i, sigma in constraints.items() if sigma != -1]

        self.constraints += constraints
        self.compiled_constraints += [CompiledOperator(c, self.gpu) for c in constraints]

    def create_multiple_spin_constraints(self, operator, N, average_length):
        my_constraints = []

        while len(my_constraints) < N:
            for term in operator:
                constraint = {}

                for i, sigma in term.pauli_string:
                    average_fraction = average_length / len(term.pauli_string)

                    if random.randint(1, int(10 / average_fraction)) > 10:
                        continue

                    constraint[i] = (sigma + random.randint(0, 1)) % 3 + 1

                if len(constraint) <= 1:
                    continue

                if len(constraint) % 2 == 1:
                    del constraint[next(iter(constraint))]

                my_constraints.append(pe({i: sigma for i, sigma in constraint.items()}))

        self.constraints += my_constraints
        self.compiled_constraints += [CompiledOperator(c, self.gpu) for c in my_constraints]
