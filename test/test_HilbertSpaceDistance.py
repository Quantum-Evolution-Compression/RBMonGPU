from pyRBMonGPU import HilbertSpaceDistance, ExactSummation, Operator
from pytest import approx
import numpy as np
from pathlib import Path
import json


def test_distance1(psi_all, hamiltonian, gpu):
    psi = psi_all(gpu)

    N = psi.N
    H = hamiltonian(N)
    spin_ensemble = ExactSummation(N, gpu)

    psi.normalize(spin_ensemble)

    t = 1e-3
    hs_distance = HilbertSpaceDistance(N, psi.num_params, gpu)
    op = Operator(1j * psi.transform(H) * t, gpu)
    distance_test = hs_distance(psi, psi, op, False, spin_ensemble)

    H_diag = np.linalg.eigh(H.matrix(N))
    U_t = H_diag[1] @ (np.exp(1j * H_diag[0] * t) * H_diag[1]).T
    psi_vector = psi.vector
    distance_ref = 1.0 - abs(np.vdot(psi_vector, U_t @ psi_vector))

    assert distance_test == approx(distance_ref, rel=1e-3, abs=1e-8)


def test_distance2(psi_all, hamiltonian, gpu):
    psi = psi_all(gpu)

    N = psi.N
    H = hamiltonian(N)
    spin_ensemble = ExactSummation(N, gpu)

    psi.normalize(spin_ensemble)
    psi1 = +psi
    psi1.params = 0.95 * psi1.params
    psi1.normalize(spin_ensemble)

    t = 1e-4
    hs_distance = HilbertSpaceDistance(N, psi.num_params, gpu)
    op = Operator(1j * psi.transform(H) * t, gpu)
    distance_test = hs_distance(psi, psi1, op, False, spin_ensemble)

    H_diag = np.linalg.eigh(H.matrix(N))
    U_t = H_diag[1] @ (np.exp(1j * H_diag[0] * t) * H_diag[1]).T
    distance_ref = 1.0 - abs(np.vdot(psi.vector, U_t @ psi1.vector))

    assert distance_test == approx(distance_ref, rel=1e-1, abs=1e-8)


def test_gradient(psi_all, hamiltonian, gpu):
    psi = psi_all(gpu)

    N = psi.N
    H = hamiltonian(N)
    spin_ensemble = ExactSummation(N, gpu)

    psi.normalize(spin_ensemble)
    psi1 = +psi

    hs_distance = HilbertSpaceDistance(N, psi.num_params, gpu)
    op = Operator(1j * psi.transform(H) * 1e-2, gpu)

    gradient_ref, distance_ref = hs_distance.gradient(psi, psi, op, False, spin_ensemble)
    gradient_ref[:2 * N] = gradient_ref[:2 * N].real

    eps = 1e-6

    def distance_diff(delta_params):
        psi1.params = psi.params + delta_params
        plus_distance = hs_distance(psi, psi1, op, False, spin_ensemble)

        psi1.params = psi.params - delta_params
        minus_distance = hs_distance(psi, psi1, op, False, spin_ensemble)

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
