from make_PsiDynamical import psi_zeros, psi_random
from pyRBMonGPU import Spins
from pytest import approx
import pytest
import numpy as np
import cmath
import random


def test_num_params(gpu):
    N = 4
    M = 8

    psi = psi_zeros(N, M, gpu)

    assert psi.num_params > N + M
    assert psi.num_params == psi.num_active_params


def test_psi_vector(gpu):
    N = 5
    M = 10

    psi = psi_zeros(N, M, gpu)
    assert psi.gpu == gpu

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


def test_get_set_params(gpu):
    N = 4
    M = 7
    noise = 0.5

    psi, a, b, first_spins, W_sparse = psi_random(N, M, noise, gpu)

    ref_vector = psi.vector

    params = psi.active_params
    psi.active_params = params

    assert np.array_equal(psi.vector, ref_vector)


def test_random_psi(gpu):
    N = 10
    M = 6
    noise = 0.1

    psi, a, b, first_spins, W_sparse = psi_random(N, M, noise, gpu)

    W = np.zeros((N, M), dtype=complex)
    for j, (first_spin, W_j) in enumerate(zip(first_spins, W_sparse)):
        begin, end = first_spin, (first_spin + len(W_j)) % N
        if end <= begin:
            W[begin:, j] += W_j[:N - begin]
            W[:end, j] += W_j[N - begin:]
        else:
            W[begin: end, j] += W_j

    target = np.zeros(2**N, dtype=complex)

    for s in range(2**N):
        spins = Spins(s).array(N)

        for h in range(2**M):
            hidden_spins = Spins(h).array(M)
            log_psi = (
                a @ spins +
                b @ hidden_spins +
                spins @ W @ hidden_spins
            )

            target[s] += cmath.exp(log_psi)

    print(psi.vector[:3])
    print(target[:3])

    assert np.allclose(psi.vector, target)


def test_O_k_vector(gpu):
    N = 10
    M = 100
    noise = 0.1

    psi, a, b, first_spins, W_sparse = psi_random(N, M, noise, gpu)

    W = np.zeros((N, M), dtype=complex)
    for j, (first_spin, W_j) in enumerate(zip(first_spins, W_sparse)):
        begin, end = first_spin, (first_spin + len(W_j)) % N
        if end <= begin:
            W[begin:, j] += W_j[:N - begin]
            W[:end, j] += W_j[N - begin:]
        else:
            W[begin: end, j] += W_j

    spins = Spins(random.randint(0, 2**N - 1))
    spins_array = spins.array(N)
    O_k_vector = psi.O_k_vector(spins)

    for i in range(N):
        assert O_k_vector[i] == spins_array[i]

    for j, b_j in enumerate(b):
        assert O_k_vector[N + j] == approx(
            cmath.tanh(W[:, j] @ spins_array + b_j)
        )

    params_offset = 0
    for j in range(M):
        W_j = W_sparse[j]

        for i in range(len(W_j)):
            assert (
                O_k_vector[N + M + params_offset + i] == approx(
                    cmath.tanh(
                        W[:, j] @ spins_array + b[j]
                    ) * spins_array[(first_spins[j] + i) % N]
                )
            )

        params_offset += len(W_j)
