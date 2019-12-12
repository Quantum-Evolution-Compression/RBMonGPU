from ._pyRBMonGPU import PsiDeep
from json_numpy import NumpyEncoder, NumpyDecoder
import json


def to_json(self):
    obj = dict(
        type="PsiDeep",
        a=self.a,
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
        obj["b"],
        obj["connections"],
        obj["W"],
        obj["prefactor"],
        gpu
    )


def normalize(self, exact_summation):
    self.prefactor /= self.norm(exact_summation)


def __pos__(self):
    return self.copy()


setattr(PsiDeep, "to_json", to_json)
setattr(PsiDeep, "from_json", from_json)
setattr(PsiDeep, "normalize", normalize)
setattr(PsiDeep, "__pos__", __pos__)
