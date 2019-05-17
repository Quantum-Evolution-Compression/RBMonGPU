from pyRBMonGPU import PsiDynamical
import numpy as np
import random


def psi_zeros(N, M, gpu):
    a = np.zeros(N, dtype=complex)
    b = np.zeros(M, dtype=complex)

    result = PsiDynamical(a, gpu)

    for i, b_spin in enumerate(b):
        result.add_hidden_spin(0, [0.0, 0.0, 0.0], b_spin)
    result.update()

    return result


def psi_random(N, M, noise, gpu):
    a = np.zeros(N, dtype=complex)
    b = np.zeros(M, dtype=complex)

    a += noise * (
        (2 * np.random.rand(N) - 1) +
        1j * (2 * np.random.rand(N) - 1)
    )

    result = PsiDynamical(a, gpu)

    b += noise * (
        (2 * np.random.rand(M) - 1) +
        1j * (2 * np.random.rand(M) - 1)
    )

    first_spins = []
    W = []

    for i, b_spin in enumerate(b):
        first_spin = random.randint(0, N - 1)
        num_spins = random.randint(1, N - 1)
        W_j = noise * (
            (2 * np.random.rand(num_spins) - 1) +
            1j * (2 * np.random.rand(num_spins) - 1)
        )
        result.add_hidden_spin(first_spin, W_j, b_spin)

        first_spins.append(first_spin)
        W.append(W_j)

    result.update()

    return result, a, b, first_spins, W
