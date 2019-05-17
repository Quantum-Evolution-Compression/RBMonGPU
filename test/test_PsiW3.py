from make_PsiW3 import psi_zeros, psi_random
from pyRBMonGPU import PsiW3, Spins
from pytest import approx
import numpy as np
import cmath
import random


def test_num_params(gpu):
    N = 4
    M = 8
    F = 4

    psi = psi_zeros(N, M, F, gpu)

    assert psi.num_params == N + M + N * M + M + N * F + F * M
    assert psi.num_active_params == N + M + N * M + N * F + F * M


def test_psi_vector(gpu):
    N = 5
    M = 10
    F = 5

    psi = psi_zeros(N, M, F, gpu)
    assert psi.gpu == gpu

    assert np.allclose(
        psi.vector,
        2**M * np.ones(2**N, dtype=complex)
    )


def test_norm(gpu):
    N = 3
    M = 9
    F = 6

    psi = psi_zeros(N, M, F, gpu)

    print(psi.norm, 2**((N + 2 * M) / 2))

    assert approx(psi.norm) == approx(2**((N + 2 * M) / 2))


def test_random_psi(gpu):
    N = 3
    M = 6
    F = 9
    noise = 0.1

    psi, a, b, W, n, X, Y = psi_random(N, M, F, noise, gpu)

    W3 = np.zeros((N, M, N), dtype=complex)
    for f in range(F):
        W3 += np.multiply.outer(np.outer(X[:, f], Y[f, :]), X[:, f])

    target = np.zeros(2**N, dtype=complex)

    for s in range(2**N):
        spins = Spins(s).array(N)

        for h in range(2**M):
            hidden_spins = Spins(h).array(M)
            log_psi = (
                a @ spins +
                b @ hidden_spins +
                spins @ W @ hidden_spins +
                spins @ (W3 @ spins) @ hidden_spins
            )

            target[s] += cmath.exp(log_psi)

    assert np.allclose(psi.vector, target)


# def test_O_k_vector(gpu):
#     N = 4
#     M = 8
#     F = 6
#     noise = 0.1
#     eps = 1e-4

#     psi, a, b, W, n, X, Y = psi_random(N, M, F, noise, gpu)
#     spins = Spins(random.randint(0, 2**N - 1))
#     spins_array = spins.array(N)
#     O_k_vector = psi.O_k_vector(spins)

#     W3 = np.zeros((N, M, N), dtype=complex)
#     for f in range(F):
#         W3 += np.multiply.outer(np.outer(X[:, f], Y[f, :]), X[:, f])

#     for i in range(N):
#         assert O_k_vector[i] == spins_array[i]

#     for j, b_j in enumerate(b):
#         assert O_k_vector[N + j] == approx(
#             cmath.tanh(
#                 spins_array @ W3[:, j, :] @ spins_array + W[:, j] @ spins_array + b_j
#             )
#         )

#     for i in range(N):
#         for j in range(M):
#             assert (
#                 O_k_vector[N + M + i * M + j] == approx(
#                     cmath.tanh(
#                         spins_array @ W3[:, j, :] @ spins_array + W[:, j] @ spins_array + b[j]
#                     ) * spins_array[i]
#                 )
#             )

#     for i in range(N):
#         for f in range(F):
#             assert O_k_vector[N + M + N * M + i * F + f] == approx(
#                 sum(
#                     cmath.tanh(
#                         spins_array @ W3[:, j, :] @ spins_array + W[:, j] @ spins_array + b[j]
#                     ) * Y[f, j] * 2 * (spins_array @ X[:, f]) * spins_array[i]
#                     for j in range(M)
#                 )
#             )

#     for j in range(M):
#         for f in range(F):
#             assert O_k_vector[N + M + N * M + N * F + f * M + j] == approx(
#                 cmath.tanh(
#                     spins_array @ W3[:, j, :] @ spins_array + W[:, j] @ spins_array + b[j]
#                 ) * (X[:, f] @ spins_array)**2
#             )


# def test_derivative(gpu):
#     N = 4
#     M = 8
#     F = 6
#     noise = 0.1
#     eps = 1e-4

#     psi, a, b, W, n, X, Y = psi_random(N, M, F, noise, gpu)
#     s = random.randint(0, 2**N - 1)
#     spins = Spins(s)
#     O_k_vector = psi.O_k_vector(spins)

#     W3 = np.zeros((N, M, N), dtype=complex)
#     for f in range(F):
#         W3 += np.multiply.outer(np.outer(X[:, f], Y[f, :]), X[:, f])

#     for i in range(N):
#         a_plus = +a
#         a_minus = +a
#         a_plus[i] += eps
#         a_minus[i] -= eps
#         psi_plus = PsiW3(a_plus, b, W, n, X, Y, 1.0, gpu)
#         psi_minus = PsiW3(a_minus, b, W, n, X, Y, 1.0, gpu)

#         diff = cmath.log(psi_plus.vector[s]) - cmath.log(psi_minus.vector[s])

#         assert O_k_vector[i] == approx(diff / (2 * eps))

#     for j in range(M):
#         b_plus = +b
#         b_minus = +b
#         b_plus[j] += eps
#         b_minus[j] -= eps
#         psi_plus = PsiW3(a, b_plus, W, n, X, Y, 1.0, gpu)
#         psi_minus = PsiW3(a, b_minus, W, n, X, Y, 1.0, gpu)

#         diff = cmath.log(psi_plus.vector[s]) - cmath.log(psi_minus.vector[s])

#         assert O_k_vector[N + j] == approx(diff / (2 * eps))

#     for i in range(N):
#         for j in range(M):
#             W_plus = +W
#             W_minus = +W
#             W_plus[i, j] += eps
#             W_minus[i, j] -= eps
#             psi_plus = PsiW3(a, b, W_plus, n, X, Y, 1.0, gpu)
#             psi_minus = PsiW3(a, b, W_minus, n, X, Y, 1.0, gpu)

#             diff = cmath.log(psi_plus.vector[s]) - cmath.log(psi_minus.vector[s])

#             assert O_k_vector[N + M + i * M + j] == approx(diff / (2 * eps))

#     for i in range(N):
#         for f in range(F):
#             X_plus = +X
#             X_minus = +X
#             X_plus[i, f] += eps
#             X_minus[i, f] -= eps
#             psi_plus = PsiW3(a, b, W, n, X_plus, Y, 1.0, gpu)
#             psi_minus = PsiW3(a, b, W, n, X_minus, Y, 1.0, gpu)

#             diff = cmath.log(psi_plus.vector[s]) - cmath.log(psi_minus.vector[s])

#             assert O_k_vector[N + M + N * M + i * F + f] == approx(diff / (2 * eps))

#     for f in range(F):
#         for j in range(M):
#             Y_plus = +Y
#             Y_minus = +Y
#             Y_plus[f, j] += eps
#             Y_minus[f, j] -= eps
#             psi_plus = PsiW3(a, b, W, n, X, Y_plus, 1.0, gpu)
#             psi_minus = PsiW3(a, b, W, n, X, Y_minus, 1.0, gpu)

#             diff = cmath.log(psi_plus.vector[s]) - cmath.log(psi_minus.vector[s])

#             assert O_k_vector[N + M + N * M + N * F + f * M + j] == approx(diff / (2 * eps))
