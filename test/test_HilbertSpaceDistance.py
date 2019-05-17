from pyRBMonGPU import HilbertSpaceDistance, ExactSummation, Spins
import pytest
from pytest import approx
from scipy.linalg import expm
import numpy as np
from copy import deepcopy


@pytest.mark.parametrize("unitary", (True, False))
def test_distance(psi, operator, gpu, unitary):
    psi = psi(gpu)[0]
    N = psi.N
    operator = operator(N, unitary)

    exact_a = psi.vector
    if unitary:
        exact_b = operator.matrix(N) @ exact_a
    else:
        exact_b = expm(operator.matrix(N)) @ exact_a

    ab = exact_a @ exact_b.conj()
    exact_distance = 1.0 - (
        abs(ab)**2 / ((exact_a @ exact_a.conj()) * (exact_b @ exact_b.conj()))
    )**0.5

    hilbert_space_distance = HilbertSpaceDistance(psi.num_params, gpu)
    spin_ensemble = ExactSummation(N)
    compiled_op = CompiledOperator(operator, gpu)

    assert hilbert_space_distance(
        psi, psi, compiled_op, unitary, spin_ensemble, False
    ) == approx(exact_distance, rel=1e-2 if unitary else 5e-2)


# @pytest.mark.parametrize("unitary", (True, False))
# def test_gradient(psi, operator, gpu, unitary):
#     eps = 1e-4

#     psi_output = psi(gpu)
#     psi = psi_output[0]
#     args = psi_output[1:]
#     psi0 = psi.__class__(*args, 1.0, gpu)

#     N = psi.N
#     M = psi.M
#     F = psi.F if hasattr(psi, "F") else None

#     hilbert_space_distance = HilbertSpaceDistance(psi.num_active_params, gpu)
#     compiled_op = CompiledOperator(operator(N, unitary), gpu)
#     spin_ensemble = ExactSummation(N)

#     my_distance = lambda psi: (
#         hilbert_space_distance(psi0, psi, compiled_op, unitary, spin_ensemble, False)
#     )

#     def my_derivative(arg_idx, idx, c=1):
#         args_plus = deepcopy(args)
#         args_minus = deepcopy(args)
#         args_plus[arg_idx][idx] += c * eps
#         args_minus[arg_idx][idx] -= c * eps

#         psi_plus = psi.__class__(*args_plus, 1.0, gpu)
#         psi_minus = psi.__class__(*args_minus, 1.0, gpu)

#         diff = my_distance(psi_plus) - my_distance(psi_minus)
#         return diff / (2 * eps)

#     target_gradient = np.zeros(psi.num_active_params, dtype=complex)

#     for i in range(N):
#         target_gradient[i] = my_derivative(0, i) + 1j * my_derivative(0, i, 1j)

#     for j in range(M):
#         target_gradient[N + j] = my_derivative(1, j) + 1j * my_derivative(1, j, 1j)

#     for i in range(N):
#         for j in range(M):
#             target_gradient[N + M + i * M + j] = my_derivative(2, (i, j)) + 1j * my_derivative(2, (i, j), 1j)

#     if F is not None:
#         for i in range(N):
#             for f in range(F):
#                 target_gradient[N + M + N * M + i * F + f] = my_derivative(4, (i, f)) + 1j * my_derivative(4, (i, f), 1j)

#         for f in range(F):
#             for j in range(M):
#                 target_gradient[N + M + N * M + N * F + f * M + j] = my_derivative(5, (f, j)) + 1j * my_derivative(5, (f, j), 1j)

#     test_gradient = hilbert_space_distance.gradient(
#         psi0, psi0, compiled_op, unitary, spin_ensemble
#     )[0]

#     # print((test_gradient - target_gradient).real / target_gradient.real)
#     # print(target_gradient)

#     assert test_gradient == approx(target_gradient, rel=3e-1)
