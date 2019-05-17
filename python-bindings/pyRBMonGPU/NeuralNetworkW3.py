import pyRBMonGPU
import numpy as np
import math


class NeuralNetworkW3(pyRBMonGPU.PsiW3):
    def __init__(
        self,
        a,
        b,
        W,
        n,
        X,
        Y,
        prefactor=None,
        spin_ensemble=None,
        gpu=True,
        mc=True
    ):
        super().__init__(a, b, W, n, X, Y, prefactor, gpu)
        self.a = a
        self.b = b
        self.W = W
        self.n = n
        self.X = X
        self.Y = Y

        self.spin_ensemble = spin_ensemble
        self.mc = mc

    def copy(self):
        result = NeuralNetworkW3(
            +self.a,
            +self.b,
            +self.W,
            +self.n,
            +self.X,
            +self.Y,
            +self.prefactor,
            self.spin_ensemble,
            self.gpu,
            self.mc
        )
        return result

    @property
    def psi(self):
        """
        CAUTION: If Monte-Carlo is enabled this vector is not yet normalized!
        """
        return self.psi_vector()

    @property
    def norm(self):
        return math.sqrt(np.sum(abs(self.psi)**2))

    @property
    def params(self):
        return np.hstack(
            (self.a, self.b, self.W.flatten(), self.n, self.X.flatten(), self.Y.flatten())
        )

    @params.setter
    def params(self, new_params):
        self.a = new_params[:self.N]
        offset = self.N
        self.b = new_params[offset: offset + self.M]
        offset += self.M
        self.W = new_params[offset: offset + self.N * self.M].reshape((self.N, self.M))
        offset += self.N * self.M
        self.n = new_params[offset: offset + self.M]
        offset += self.M
        self.X = new_params[offset: offset + self.N * self.F].reshape((self.N, self.F))
        offset += self.N * self.F
        self.Y = new_params[offset:].reshape((self.F, self.M))

        self.update()

    @property
    def active_params(self):
        result = np.hstack((self.a, self.b, self.W.flatten(), self.X.flatten(), self.Y.flatten()))
        return result

    @active_params.setter
    def active_params(self, new_active_params):
        self.a = new_active_params[:self.N]
        offset = self.N
        self.b = new_active_params[offset: offset + self.M]
        offset += self.M
        self.W = new_active_params[offset: offset + self.N * self.M].reshape((self.N, self.M))
        offset += self.N * self.M
        self.X = new_active_params[offset: offset + self.N * self.F].reshape((self.N, self.F))
        offset += self.N * self.F
        self.Y = new_active_params[offset:].reshape((self.F, self.M))

        self.update()

    def update(self):
        self.update_params(self.a, self.b, self.W, self.n, self.X, self.Y)

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

    # def observable_over_mc_steps(self, operator, num_sweeps):
    #     observable_history = self.spin_ensemble.observable_history(
    #         self.psi_gpu, CompiledOperator(operator, self.gpu), 0, num_sweeps
    #     )

    #     return np.mean(observable_history, axis=1)

    # def autocorrelation_function(self, operator, num_sweeps, num_thermalization_sweeps=None):
    #     observable_history = self.spin_ensemble.observable_history(
    #         self.psi_gpu, CompiledOperator(operator, self.gpu), num_sweeps, num_thermalization_sweeps or 0
    #     )

    #     observable_0 = observable_history[0, :]

    #     normalization = np.mean(abs(observable_0)**2) - abs(np.mean(observable_0))**2

    #     return (
    #         np.mean(observable_0[np.newaxis, :] * observable_history.conj(), axis=1) -
    #         np.mean(observable_0) * np.mean(observable_history, axis=1).conj()
    #     ) / normalization
