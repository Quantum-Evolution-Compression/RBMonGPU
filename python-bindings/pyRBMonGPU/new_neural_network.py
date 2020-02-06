# from pyRBMonGPU import Psi, PsiDeep
from pyRBMonGPU import PsiDeep
import numpy as np
import math
from itertools import product


def real_noise(shape):
    return 2 * np.random.random_sample(shape) - 1


def complex_noise(shape):
    return real_noise(shape) + 1j * real_noise(shape)


def new_neural_network(
    N,
    M,
    initial_value=(0.01 + 1j * math.pi / 4),
    alpha=0,
    beta=0,
    free_quantum_axis=False,
    noise=1e-6,
    gpu=False
):
    if isinstance(alpha, (int, float)):
        alpha = alpha * np.ones(N)
    if isinstance(beta, (int, float)):
        beta = beta * np.ones(N)

    b = noise * complex_noise(M)
    W = noise * complex_noise((N, M))

    for r in range(M // N):
        W[:, r * N:(r + 1) * N] += initial_value * np.diag(np.ones(N))

    return Psi(alpha, beta, b, W, 1, free_quantum_axis, gpu)


def new_deep_neural_network(
    N,
    M,
    C,
    dim=1,
    initial_value=(0.01 + 1j * math.pi / 4),
    alpha=0,
    beta=0,
    free_quantum_axis=False,
    noise=1e-4,
    translational_invariance=False,
    gpu=False
):
    N_linear = N if dim == 1 else N[0] * N[1]
    M_linear = M if dim == 1 else [m[0] * m[1] for m in M]
    C_linear = C if dim == 1 else [c[0] * c[1] for c in C]

    for n, m, c in zip([N] + M[:-1], M, C):
        if dim == 1:
            assert m * c % n == 0
            assert c <= n
        elif dim == 2:
            assert m[0] * c[0] % n[0] == 0
            assert m[1] * c[1] % n[1] == 0
            assert c[0] <= n[0]
            assert c[1] <= n[1]

    if isinstance(alpha, (float, int)):
        alpha = alpha * np.ones(N_linear)
    if isinstance(beta, (float, int)):
        beta = beta * np.ones(N_linear)

    b = [noise * complex_noise(m) for m in M_linear]

    w = noise * complex_noise((C_linear[0], M_linear[0]))

    if translational_invariance:
        w = noise * np.stack([complex_noise((M_linear[0]))] * N_linear, axis=0)

    w[C_linear[0] // 2, :] += initial_value
    W = [w]

    for c, m, next_c in zip(C_linear[1:], M_linear[1:], C_linear[2:] + [1]):
        w = (
            math.sqrt(6 / (c + next_c)) * real_noise((c, m)) +
            # 1j * math.sqrt(6 / (c + next_c)) / 1e2 * real_noise((c, m)) +
            noise * complex_noise((c, m))
        )
        W.append(w)

    def delta_func(n, m, c):
        if m > n:
            return 1
        elif n % m == 0:
            return n // m
        else:
            return c

    connections = []
    for n, m, c in zip([N] + M[:-1], M, C):
        if dim == 1:
            delta_j = delta_func(n, m, c)

            connections.append(np.array([
                [
                    (j * delta_j + i) % n
                    for j in range(m)
                ]
                for i in range(c)
            ]))
        elif dim == 2:
            range2D = lambda area: product(range(area[0]), range(area[1]))
            linear_idx = lambda row, col: row * n[0] + col

            delta_j1 = delta_func(n[0], m[0], c[0])
            delta_j2 = delta_func(n[1], m[1], c[1])

            connections.append(np.array([
                [
                    linear_idx(
                        (j1 * delta_j1 + i1) % n[0],
                        (j2 * delta_j2 + i2) % n[1]
                    )
                    for j1, j2 in range2D(m)
                ]
                for i1, i2 in range2D(c)
            ]))

    return PsiDeep(alpha, beta, b, connections, W, 1.0, free_quantum_axis, gpu)
