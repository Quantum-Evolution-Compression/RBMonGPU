from pyRBMonGPU import HilbertSpaceDistance, ExactSummation, Operator
import numpy as np
from pathlib import Path
import json


def test_gradient(psi_all, hamiltonian, gpu):
    psi = psi_all(gpu)

    N = psi.N
    H = hamiltonian(N)
    spin_ensemble = ExactSummation(N, gpu)

    psi.normalize(spin_ensemble)
    psi1 = +psi

    hs_distance = HilbertSpaceDistance(N, psi.num_params, gpu)
    op = Operator(1 + 1j * psi.transform(H) * 1e-2, gpu)

    gradient_ref, distance_ref = hs_distance.gradient(psi, psi, op, True, spin_ensemble)
    gradient_ref[:2 * N] = gradient_ref[:2 * N].real

    eps = 1e-6

    def distance_diff(delta_params):
        psi1.params = psi.params + delta_params
        plus_distance = hs_distance(psi, psi1, op, True, spin_ensemble)

        psi1.params = psi.params - delta_params
        minus_distance = hs_distance(psi, psi1, op, True, spin_ensemble)

        return (plus_distance - minus_distance) / (2 * eps)

    gradient_test = np.zeros(psi.num_params, dtype=complex)

    for k in range(psi.num_params):
        delta_params = np.zeros(psi.num_params, dtype=complex)
        delta_params[k] = eps
        gradient_test[k] = distance_diff(delta_params)

        if k >= 2 * N:
            delta_params = np.zeros(psi.num_params, dtype=complex)
            delta_params[k] = 1j * eps
            gradient_test[k] += 1j * distance_diff(delta_params)

    print(gradient_test - gradient_ref)
    print(gradient_ref)

    condition = np.allclose(gradient_test, gradient_ref, rtol=1e-2, atol=1e-8)

    if not condition:
        with open(Path().home() / "test_gradient.json", "w") as f:
            json.dump(
                {
                    "gradient_ref.real": gradient_ref.real.tolist(),
                    "gradient_ref.imag": gradient_ref.imag.tolist(),
                    "gradient_test.real": gradient_test.real.tolist(),
                    "gradient_test.imag": gradient_test.imag.tolist(),
                },
                f
            )

    assert condition
