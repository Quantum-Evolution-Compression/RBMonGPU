from pyRBMonGPU import Psi
import numpy as np


def psi_zeros(N, M, gpu):
    a = np.zeros(N, dtype=complex)
    b = np.zeros(M, dtype=complex)
    W = np.zeros((N, M), dtype=complex)
    n = np.ones(M, dtype=complex)

    return Psi(a, b, W, n, 1.0, gpu)


def psi_random(N, M, noise, gpu):
    a = np.zeros(N, dtype=complex)
    b = np.zeros(M, dtype=complex)
    W = np.zeros((N, M), dtype=complex)
    n = np.ones(M, dtype=complex)

    a += noise * (
        (2 * np.random.rand(N) - 1) +
        1j * (2 * np.random.rand(N) - 1)
    )
    b += noise * (
        (2 * np.random.rand(M) - 1) +
        1j * (2 * np.random.rand(M) - 1)
    )
    W += noise * (
        (2 * np.random.rand(N, M) - 1) +
        1j * (2 * np.random.rand(N, M) - 1)
    )

    return Psi(a, b, W, n, 1.0, gpu), a, b, W, n
