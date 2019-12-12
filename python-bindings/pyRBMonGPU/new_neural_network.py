from pyRBMonGPU import PsiDynamical, Psi, PsiDeep
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
    connectivity=5,
    noise=1e-6,
    gpu=False
):
    a = noise * complex_noise(N)
    b = noise * complex_noise(M)

    result = PsiDynamical(a, gpu)

    for j, b_spin in enumerate(b):
        first_spin = (j - connectivity // 2 + N) % N
        W_j = noise * complex_noise(connectivity)
        W_j[connectivity // 2] = initial_value
        result.add_hidden_spin(first_spin, W_j, b_spin)

    result.update()

    return result


def add_hidden_spin(psi, position, connectivity, initial_value=(0.01 + 1j * math.pi / 4), noise=1e-6):
    W_j = noise * complex_noise(connectivity)
    W_j[connectivity // 2] += initial_value
    psi.add_hidden_spin((position - connectivity // 2 + psi.N) % psi.N, W_j, noise * complex_noise(1)[0])


def add_hidden_layer(psi, connectivity, initial_value=(0.01 + 1j * math.pi / 4), noise=1e-6):
    for i in range(psi.N):
        add_hidden_spin(psi, i, connectivity, initial_value, noise)


def new_static_neural_network(
    N,
    M,
    initial_value=(0.01 + 1j * math.pi / 4),
    noise=1e-6,
    gpu=False
):
    a = noise * complex_noise(N)
    b = noise * complex_noise(M)
    W = noise * complex_noise((N, M))

    for alpha in range(M // N):
        W[:, alpha * N:(alpha + 1) * N] += initial_value * np.diag(np.ones(N))

    return Psi(a, b, W, 1, gpu)


def new_deep_neural_network(
    N,
    M,
    C,
    dim=1,
    initial_value=(0.01 + 1j * math.pi / 4),
    noise=1e-4,
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

    a = noise * complex_noise(N_linear)
    b = [noise * complex_noise(m) for m in M_linear]

    w = noise * complex_noise((C_linear[0], M_linear[0]))

    w[C_linear[0] // 2, :] = initial_value
    W = [w]

    for c, m, next_c in zip(C_linear[1:], M_linear[1:], C_linear[2:] + [1]):
        w = (
            math.sqrt(6 / (c + next_c)) * real_noise((c, m)) +
            # 1j * math.sqrt(6 / (c + next_c)) / 1e2 * real_noise((c, m)) +
            noise * complex_noise((c, m))
        )
        W.append(w)

    connections = []
    for n, m, c in zip([N] + M[:-1], M, C):
        if dim == 1:
            delta_j = m * c / n
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

            delta_j1 = m[0] * c[0] / n[0]
            delta_j2 = m[1] * c[1] / n[1]
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

    return PsiDeep(a, b, connections, W, 1.0, gpu)
