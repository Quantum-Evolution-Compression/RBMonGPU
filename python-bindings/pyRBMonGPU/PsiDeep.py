from ._pyRBMonGPU import PsiDeep
from json_numpy import NumpyEncoder, NumpyDecoder
from QuantumExpression import sigma_x, sigma_y
import json


def to_json(self):
    obj = dict(
        type="PsiDeep",
        alpha=self.alpha,
        beta=self.beta,
        b=self.b,
        connections=self.connections,
        W=self.W,
        prefactor=self.prefactor,
        free_quantum_axis=self.free_quantum_axis
    )

    return json.loads(
        json.dumps(obj, cls=NumpyEncoder)
    )


@staticmethod
def from_json(json_obj, gpu):
    obj = json.loads(
        json.dumps(
            json_obj,
            cls=NumpyEncoder
        ),
        cls=NumpyDecoder
    )

    return PsiDeep(
        obj["alpha"],
        obj["beta"],
        obj["b"],
        obj["connections"],
        obj["W"],
        obj["prefactor"],
        obj["free_quantum_axis"],
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


setattr(PsiDeep, "to_json", to_json)
setattr(PsiDeep, "from_json", from_json)
setattr(PsiDeep, "transform", transform)
setattr(PsiDeep, "normalize", normalize)
setattr(PsiDeep, "__pos__", __pos__)
setattr(PsiDeep, "vector", vector)
