from pyRBMonGPU import Operator
import numpy as np
import json


class OperatorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Operator):
            coefficients = obj.coefficients

            return {
                "type": "Operator",
                "coefficients.real": coefficients.real.tolist(),
                "coefficients.imag": coefficients.imag.tolist(),
                "pauli_types": obj.pauli_types.tolist(),
                "pauli_indices": obj.pauli_indices.tolist()
            }

        return super().default(obj)


class OperatorDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        self.gpu = kwargs["gpu"] if "gpu" in kwargs else False

        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if "type" in obj and obj["type"] == "Operator":
            coefficients = np.array(obj["coefficients.real"]) + 1j * np.array(obj["coefficients.imag"])
            return Operator(
                coefficients,
                np.array(obj["pauli_types"]),
                np.array(obj["pauli_indices"]),
                self.gpu
            )

        return obj
