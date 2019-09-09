from ._pyRBMonGPU import PsiDynamical
from json_complex import ComplexEncoder, ComplexDecoder
import json


def to_json(self):
    links = [
        dict(
            first_spin=link.first_spin,
            weights=link.weights,
            hidden_spin_weight=link.hidden_spin_weight,
            hidden_spin_type=link.hidden_spin_type
        ) for link in self.links
    ]

    obj = dict(
        type="PsiDynamical",
        prefactor=self.prefactor,
        spin_weights=self.spin_weights,
        links=links
    )

    return json.loads(
        json.dumps(obj, cls=ComplexEncoder)
    )


@staticmethod
def from_json(json_obj, gpu):
    obj = json.loads(
        json.dumps(json_obj),
        cls=ComplexDecoder
    )

    result = PsiDynamical(obj["spin_weights"], gpu)
    result.prefactor = obj["prefactor"]
    for link in obj["links"]:
        result.add_hidden_spin(
            link["first_spin"],
            link["weights"],
            link["hidden_spin_weight"],
            link["hidden_spin_type"]
        )
    result.update()
    return result


def normalize(self):
    self.prefactor /= self.norm


def __pos__(self):
    return self.copy()


setattr(PsiDynamical, "to_json", to_json)
setattr(PsiDynamical, "from_json", from_json)
setattr(PsiDynamical, "normalize", normalize)
setattr(PsiDynamical, "__pos__", __pos__)
