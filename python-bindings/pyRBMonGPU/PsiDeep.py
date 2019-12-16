from ._pyRBMonGPU import PsiDeep
from json_numpy import NumpyEncoder, NumpyDecoder
from QuantumExpression import sigma_y
import json


def to_json(self):
    obj = dict(
        type="PsiDeep",
        a=self.a,
        alpha=self.alpha,
        b=self.b,
        connections=self.connections,
        W=self.W,
        prefactor=self.prefactor
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
        obj["a"],
        obj["alpha"],
        obj["b"],
        obj["connections"],
        obj["W"],
        obj["prefactor"],
        gpu
    )


def transform(self, operator, threshold=1e-10):
    generator = sum(-1j * alpha_i * sigma_y(i) for i, alpha_i in enumerate(self.alpha))
    return operator.rotate_by(generator, 0).apply_threshold(threshold)


def normalize(self, exact_summation):
    self.prefactor /= self.norm(exact_summation)


def __pos__(self):
    return self.copy()


setattr(PsiDeep, "to_json", to_json)
setattr(PsiDeep, "from_json", from_json)
setattr(PsiDeep, "transform", transform)
setattr(PsiDeep, "normalize", normalize)
setattr(PsiDeep, "__pos__", __pos__)
