from pyRBMonGPU import Spins, activation_function
from pytest import approx
import cmath
import random


def test_psi_s(psi, gpu):
    psi = psi(gpu)
    psi_vector = psi.vector

    b = psi.b
    W = psi.W

    N = psi.N
    M = len(b)

    for n in range(10):
        spins_idx = random.randint(0, 2**N - 1)
        spins = Spins(spins_idx).array(N)

        angles = [
            W[:, j] @ spins + b[j]
            for j in range(M)
        ]
        psi_s_ref = cmath.exp(sum(
            activation_function(angles[j])
            for j in range(M)
        ))

        assert psi_vector[spins_idx] == approx(psi_s_ref)
