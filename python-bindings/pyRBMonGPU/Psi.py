from ._pyRBMonGPU import Psi
from json_numpy import NumpyEncoder, NumpyDecoder
import json


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


@staticmethod
def from_json(json_obj, gpu):
    obj = json.loads(
        json.dumps(json_obj),
        cls=NumpyDecoder
    )

    return Psi(obj["a"], obj["b"], obj["W"], obj["prefactor"], gpu)


def normalize(self, exact_summation):
    self.prefactor /= self.norm(exact_summation)


def __pos__(self):
    return self.copy()


setattr(Psi, "to_json", to_json)
setattr(Psi, "from_json", from_json)
setattr(Psi, "normalize", normalize)
setattr(Psi, "__pos__", __pos__)
