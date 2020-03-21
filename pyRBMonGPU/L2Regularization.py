class L2Regularization:
    def __init__(self, lambda0, gamma):
        self.lambda0 = lambda0
        self.gamma = gamma

    def lambda_(self, step):
        return self.lambda0 * self.gamma**step

    def gradient(self, step, params):
        params_real = params.real
        params_real[abs(params_real) < 1e-3] = 0

        return self.lambda_(step) * 2 * params_real
