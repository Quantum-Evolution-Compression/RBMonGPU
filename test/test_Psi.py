import sys
sys.path.insert(0, '/home/burau/projects/RBMonGPU/python-bindings/build/lib.linux-x86_64-3.6')

from make_Psi import psi_zeros, psi_random
from pyRBMonGPU import Psi, Spins
from pytest import approx
import pytest
import numpy as np
import cmath
import random


def my_tanh(z):
    return 2 * z / (1 + z**2)


def test_num_params(gpu):
    N = 4
    M = 8

    psi = psi_zeros(N, M, gpu)

    assert psi.num_params == N + M + N * M + M
    assert psi.num_active_params == N + M + N * M


def test_psi_vector(gpu):
    N = 5
    M = 10

    psi = psi_zeros(N, M, gpu)
    assert psi.gpu == gpu

    print(psi.vector)

    assert np.allclose(
        psi.vector,
        2**M * np.ones(2**N, dtype=complex)
    )


def test_norm(gpu):
    N = 3
    M = 9

    psi = psi_zeros(N, M, gpu)

    print(psi.norm, 2**((N + 2 * M) / 2))

    assert approx(psi.norm) == approx(2**((N + 2 * M) / 2))


# @pytest.mark.parametrize("N", (5, 10))
# def test_random_psi(gpu, N=5):
#     M = 6
#     noise = 0.1

#     psi, a, b, W, n = psi_random(N, M, noise, gpu)

#     target = np.zeros(2**N, dtype=complex)

#     for s in range(2**N):
#         spins = Spins(s).array(N)

#         for h in range(2**M):
#             hidden_spins = Spins(h).array(M)
#             log_psi = (
#                 a @ spins +
#                 b @ hidden_spins +
#                 spins @ W @ hidden_spins
#             )

#             target[s] += cmath.exp(log_psi)

#     print(psi.vector[:3])
#     print(target[:3])

#     assert np.allclose(psi.vector, target)


def test_O_k_vector(gpu):
    N = 10
    M = 12
    noise = 0.1
    eps = 1e-4

    psi, a, b, W, n = psi_random(N, M, noise, gpu)
    spins = Spins(random.randint(0, 2**N - 1))
    spins_array = spins.array(N)
    O_k_vector = psi.O_k_vector(spins)

    for i, a_i in enumerate(a):
        a_plus = +a
        a_minus = +a
        a_plus[i] += eps
        a_minus[i] -= eps

        psi_plus = Psi(a_plus, b, W, n, 1.0, gpu)
        psi_minus = Psi(a_minus, b, W, n, 1.0, gpu)

        diff = psi_plus.log_psi(spins) - psi_minus.log_psi(spins)

        assert O_k_vector[i] == approx(diff / (2 * eps))
        assert O_k_vector[i] == spins_array[i]

    for j, b_j in enumerate(b):
        b_plus = +b
        b_minus = +b
        b_plus[j] += eps
        b_minus[j] -= eps

        psi_plus = Psi(a, b_plus, W, n, 1.0, gpu)
        psi_minus = Psi(a, b_minus, W, n, 1.0, gpu)

        diff = psi_plus.log_psi(spins) - psi_minus.log_psi(spins)

        assert O_k_vector[N + j] == approx(diff / (2 * eps))
        assert O_k_vector[N + j] == approx(my_tanh(W[:, j] @ spins_array + b_j))

    for i in range(N):
        for j in range(M):
            W_plus = +W
            W_minus = +W
            W_plus[i, j] += eps
            W_minus[i, j] -= eps

            psi_plus = Psi(a, b, W_plus, n, 1.0, gpu)
            psi_minus = Psi(a, b, W_minus, n, 1.0, gpu)

            diff = psi_plus.log_psi(spins) - psi_minus.log_psi(spins)

            assert O_k_vector[N + M + i * M + j] == approx(diff / (2 * eps))
            assert (
                O_k_vector[N + M + i * M + j] == approx(
                    my_tanh(W[:, j] @ spins_array + b[j]) * spins_array[i]
                )
            )
