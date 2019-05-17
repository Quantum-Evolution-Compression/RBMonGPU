from pyRBMonGPU import PsiW3
import numpy as np


def psi_zeros(N, M, F, gpu):
    a = np.zeros(N, dtype=complex)
    b = np.zeros(M, dtype=complex)
    W = np.zeros((N, M), dtype=complex)
    n = np.ones(M, dtype=complex)
    X = np.zeros((N, F), dtype=complex)
    Y = np.zeros((F, M), dtype=complex)

    return PsiW3(a, b, W, n, X, Y, 1.0, gpu)


def psi_random(N, M, F, noise, gpu):
    a = np.zeros(N, dtype=complex)
    b = np.zeros(M, dtype=complex)
    W = np.zeros((N, M), dtype=complex)
    n = np.ones(M, dtype=complex)
    X = np.zeros((N, F), dtype=complex)
    Y = np.zeros((F, M), dtype=complex)

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
    X += noise**0.33 * (
        (2 * np.random.rand(N, F) - 1) +
        1j * (2 * np.random.rand(N, F) - 1)
    )
    Y += noise**0.66 * (
        (2 * np.random.rand(F, M) - 1) +
        1j * (2 * np.random.rand(F, M) - 1)
    )

    return PsiW3(a, b, W, n, X, Y, 1.0, gpu), a, b, W, n, X, Y
