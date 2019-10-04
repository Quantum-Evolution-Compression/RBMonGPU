from ._pyRBMonGPU import Psi
from json_numpy import NumpyEncoder, NumpyDecoder
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


def normalize(self, exact_summation):
    self.prefactor /= self.norm(exact_summation)


def __pos__(self):
    return self.copy()


setattr(Psi, "to_json", to_json)
setattr(Psi, "from_json", from_json)
setattr(Psi, "normalize", normalize)
setattr(Psi, "__pos__", __pos__)
