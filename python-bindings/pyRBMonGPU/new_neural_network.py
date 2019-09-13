from pyRBMonGPU import PsiDynamical, Psi
import numpy as np
import math


def complex_noise(shape):
    return (2 * np.random.random_sample(shape) - 1) + 1j * (2 * np.random.random_sample(shape) - 1)


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
