import numpy as np


class L2Regularization:
    def __init__(self, lambda0, gamma, psi0):
        self.lambda0 = lambda0
        self.gamma = gamma

        self.reference_complexity = self.complexity(psi0)

    def complexity(self, psi):
        params = psi.params.real
        params[abs(params) < 4e-3] = 0
        return np.sum(params**2)

    def relative_complexity(self, psi):
        return self.complexity(psi) / self.reference_complexity

    def lambda_(self, step):
        return self.lambda0 * self.gamma**step

    def gradient(self, step, psi):
        params = psi.params.real
        params[abs(params) < 4e-3] = 0

        x = self.relative_complexity(psi) - 1

        return (
            self.lambda_(step) * (1 if x > 0 else -1 / 10) * 2 * params
        )

    def cost(self, step, psi):
        x = self.relative_complexity(psi) - 1
        return self.lambda_(step) * (x if x > 0 else -x / 10)
