from pyRBMonGPU import PsiDynamical, Psi, PsiDeep
import numpy as np
import math


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
    initial_value=(0.01 + 1j * math.pi / 4),
    noise=1e-4,
    gpu=False
):
    for n, m, c in zip([N] + M[:-1], M, C):
        assert m * c % n == 0
        assert c <= n

    a = noise * complex_noise(N)
    b = [noise * complex_noise(m) for m in M]

    w = noise * complex_noise((C[0], M[0]))
    w[C[0] // 2, :] = initial_value
    W = [w]

    for c, m, next_c in zip(C[1:], M[1:], C[2:] + [1]):
        w = math.sqrt(6 / (c + next_c)) * real_noise((c, m)) + noise * complex_noise((c, m))
        W.append(w)

    return PsiDeep(a, b, W, 1.0, gpu)
