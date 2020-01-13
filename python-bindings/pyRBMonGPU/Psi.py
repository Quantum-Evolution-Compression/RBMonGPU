from ._pyRBMonGPU import Psi
from json_numpy import NumpyEncoder, NumpyDecoder
from QuantumExpression import sigma_x, sigma_y
import json
import numpy as np


def to_json(self):
    obj = dict(
        type="Psi",
        a=self.a,
        b=self.b,
        W=self.W,
        prefactor=self.prefactor
    )

    return json.loads(
        json.dumps(obj, cls=NumpyEncoder)
    )


def json_as_numpy_array(obj):
    return json.loads(
        json.dumps(obj),
        cls=NumpyDecoder
    )


def load_array_from_json(obj):
    if isinstance(obj, np.ndarray):
        return obj
    return json_as_numpy_array(obj)


@staticmethod
def from_json(json_obj, gpu):
    return Psi(
        load_array_from_json(json_obj["a"]),
        load_array_from_json(json_obj["b"]),
        load_array_from_json(json_obj["W"]),
        json_obj["prefactor"],
        gpu
    )


def transform(self, operator, threshold=1e-10):
    alpha_generator = sum(-1j * alpha_i * sigma_y(i) for i, alpha_i in enumerate(self.alpha))
    beta_generator = sum(-1j * beta_i * sigma_x(i) for i, beta_i in enumerate(self.beta))
    return operator.rotate_by(beta_generator, 0).rotate_by(alpha_generator, 0).apply_threshold(threshold)


def normalize(self, exact_summation):
    self.prefactor /= self.norm(exact_summation)


def __pos__(self):
    return self.copy()


@property
def vector(self):
    alpha = self.alpha
    beta = self.beta

    result = self._vector
    for i in range(self.N):
        if abs(alpha[i]) > 1e-10:
            result = (1j * alpha[i] * sigma_y(i)).exp(0).sparse_matrix(self.N) @ result
        if abs(beta[i]) > 1e-10:
            result = (1j * beta[i] * sigma_x(i)).exp(0).sparse_matrix(self.N) @ result

    return result


setattr(Psi, "to_json", to_json)
setattr(Psi, "from_json", from_json)
setattr(Psi, "transform", transform)
setattr(Psi, "normalize", normalize)
setattr(Psi, "__pos__", __pos__)
setattr(Psi, "vector", vector)
