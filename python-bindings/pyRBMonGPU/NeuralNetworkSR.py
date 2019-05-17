import numpy as np
from pathlib import Path
# from .NeuralNetwork import NeuralNetwork
from scipy.integrate import ode


def pinv(matrix, rcond=0):
    try:
        U, s, Vh = svd(matrix, full_matrices=False, check_finite=False, lapack_driver="gesdd")
    except np.linalg.LinAlgError:
        matrix_path = Path("/home/burau/failed_divide_and_conquer_matrix.npy")
        if not matrix_path.exists():
            np.save(matrix_path, matrix)

        try:
            U, s, Vh = svd(matrix, full_matrices=False, check_finite=False, lapack_driver="gesvd")
        except np.linalg.LinAlgError as e:
            matrix_path = Path("/home/burau/failed_general_rectangular_matrix.npy")
            if not matrix_path.exists():
                np.save(matrix_path, matrix)
            raise e

    rcond_holds = s > rcond
    s[rcond_holds] = 1 / s[rcond_holds]
    s[~rcond_holds] = 0

    return (Vh.T @ np.diag(s) @ U.T).conjugate()


def rk4(f, t, y, delta_t):
    k_1 = f(t, y)
    k_2 = f(t + delta_t / 2, y + delta_t / 2 * k_1)
    k_3 = f(t + delta_t / 2, y + delta_t / 2 * k_2)
    k_4 = f(t + delta_t, y + delta_t * k_3)

    return y + delta_t / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)


class NeuralNetworkSR:
    def __init__(self, differentiate_psi, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)

        self.differentiate_psi = differentiate_psi
        self.history = []

    def copy(self):
        result = super(self.__class__, self).copy()
        result.__class__ = self.__class__
        result.differentiate_psi = self.differentiate_psi
        return result

    def differentiate(self, t, y, operator, rcond=None, Ax_rtol=None):
        self.params = y[:self.num_params] + 1j * y[self.num_params:]
        self.history.append(self.params)

        result = self.differentiate_psi(self, operator, self.spin_ensemble, Ax_rtol)

        return np.concatenate((result.real, result.imag))

    def integrate_step(
        self,
        operator,
        delta_t,
        rcond=1e-8,
        Ax_rtol=1e-6,
        method="euler",
        atol=1e-3,
        rtol=1e-3,
        first_step_normalized=1e-3,
        max_step_normalized=1e-2,
        compiled_operator=None,
        t=0.0
    ):
        """
        for applying exp(-1j * H) call with operator=-1j * H
        """

        params = self.params
        y0 = np.concatenate((params.real, params.imag))
        self.history = []

        compiled_operator = compiled_operator or CompiledOperator(operator, self.gpu)
        first_step = min(delta_t, first_step_normalized / operator.max_norm)
        max_step = min(delta_t, max_step_normalized / operator.max_norm)
        steps = list(np.arange(0, delta_t, first_step)) + [delta_t]

        if method == "euler":
            for step, next_step in zip(steps, steps[1:]):
                delta_t = next_step - step
                self.delta_t = delta_t
                y0 += delta_t * self.differentiate(t + step, y0, compiled_operator, rcond, Ax_rtol)
            y1 = y0
        elif method == "rk4":
            for step, next_step in zip(steps, steps[1:]):
                y0 = rk4(
                    lambda t, y: self.differentiate(t + step, y, compiled_operator, rcond, Ax_rtol),
                    t,
                    y0,
                    delta_t=next_step - step
                )
            y1 = y0
        else:
            for attempt in range(3):
                first_step_attempt = first_step / 4**attempt
                solver = ode(lambda t, y: self.differentiate(t, y, compiled_operator, rcond, Ax_rtol))
                solver.set_integrator(
                    method, atol=atol, rtol=rtol, first_step=first_step_attempt, max_step=max_step, nsteps=int(1 / first_step_attempt)
                )
                solver.set_initial_value(y0, t)
                y1 = solver.integrate(delta_t)
                if solver.successful():
                    break
                else:
                    self.params = params

            assert solver.successful(), "Integrator was not successful."

        self.params = y1[:self.num_params] + 1j * y1[self.num_params:]

        if not self.mc:
            self.prefactor /= self.norm
