import numpy as np
import math
import pyRBMonGPU


class NeuralNetwork(pyRBMonGPU.Psi):
    def __init__(
        self,
        a,
        b,
        W,
        n,
        prefactor=None,
        spin_ensemble=None,
        gpu=True,
        mc=True
    ):
        super().__init__(a, b, W, n, prefactor, gpu)
        self.a = a
        self.b = b
        self.W = W
        self.n = n

        self.spin_ensemble = spin_ensemble
        self.mc = mc

        # self.N = len(a)
        # self.M = len(b)
        # self.num_active_params = self.N + self.M + self.N * self.M
        # self.N_params = self.num_active_params + self.M

    def copy(self):
        result = NeuralNetwork(
            +self.a,
            +self.b,
            +self.W,
            +self.n,
            +self.prefactor,
            self.spin_ensemble,
            self.gpu,
            self.mc
        )
        return result

    @property
    def params(self):
        return np.hstack((self.a, self.b, self.W.flatten(), self.n))

    @params.setter
    def params(self, new_params):
        self.a = new_params[:self.N]
        self.b = new_params[self.N:self.N + self.M]
        self.W = new_params[self.N + self.M:self.num_active_params].reshape((self.N, self.M))
        self.n = new_params[self.num_active_params:]

        self.update()

    @property
    def active_params(self):
        return np.hstack((self.a, self.b, self.W.flatten()))

    @active_params.setter
    def active_params(self, new_active_params):
        self.a = new_active_params[:self.N]
        self.b = new_active_params[self.N:self.N + self.M]
        self.W = new_active_params[self.N + self.M:self.num_active_params].reshape((self.N, self.M))

        self.update()

    def update(self):
        self.update_params(self.a, self.b, self.W, self.n)

    def expectation_value(self, operator, spin_ensemble):
        if isinstance(operator, (list, tuple)):
            operator = [CompiledOperator(op, self.gpu) if not isinstance(op, pyRBMonGPU.Operator) else op for op in operator]
        elif not isinstance(operator, pyRBMonGPU.Operator):
            operator = CompiledOperator(operator, self.gpu)

        expectation_value = pyRBMonGPU.ExpectationValue(self.gpu)
        return expectation_value(
            self,
            operator,
            spin_ensemble
        )

    def observable_over_mc_steps(self, operator, num_sweeps):
        observable_history = self.spin_ensemble.observable_history(
            self.psi_gpu, CompiledOperator(operator, self.gpu), 0, num_sweeps
        )

        return np.mean(observable_history, axis=1)

    def autocorrelation_function(self, operator, num_sweeps, num_thermalization_sweeps=None):
        observable_history = self.spin_ensemble.observable_history(
            self.psi_gpu, CompiledOperator(operator, self.gpu), num_sweeps, num_thermalization_sweeps or 0
        )

        observable_0 = observable_history[0, :]

        normalization = np.mean(abs(observable_0)**2) - abs(np.mean(observable_0))**2

        return (
            np.mean(observable_0[np.newaxis, :] * observable_history.conj(), axis=1) -
            np.mean(observable_0) * np.mean(observable_history, axis=1).conj()
        ) / normalization


def fubini_study(psi, phi):
    psi_dot_phi = psi.conj() @ phi
    x = (psi_dot_phi * psi_dot_phi.conj() / ((psi.conj() @ psi) * (phi.conj() @ phi))).real
    x = min(x, 1.0)

    return math.acos(x)
