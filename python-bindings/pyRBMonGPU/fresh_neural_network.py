from .NeuralNetwork import NeuralNetwork
from .NeuralNetworkW3 import NeuralNetworkW3
from .NeuralNetworkGD import NeuralNetworkGD
from .NeuralNetworkSR import NeuralNetworkSR
from pyRBMonGPU import PsiDynamical
import pyRBMonGPU
import numpy as np
import math


class MonteCarloConfiguration:
    def __init__(self, num_samples, num_sweeps, num_thermalization_sweeps, num_markov_chains):
        self.num_samples = num_samples
        self.num_sweeps = num_sweeps
        self.num_thermalization_sweeps = num_thermalization_sweeps
        self.num_markov_chains = num_markov_chains


def complex_noise(L):
    return (2 * np.random.rand(L) - 1) + 1j * (2 * np.random.rand(L) - 1)


def new_neural_network(
    N,
    M,
    initial_value=(0.01 + 1j * math.pi / 4),
    connectivity=5,
    noise=1e-6,
    normalize=False,
    gpu=False,
):
    a = noise * complex_noise(N)
    b = noise * complex_noise(M)

    result = PsiDynamical(a, gpu)

    for j, b_spin in enumerate(b):
        first_spin = (j - connectivity // 2 + N) % N
        W_j = noise * complex_noise(connectivity)
        W_j[connectivity // 2] = initial_value
        result.add_hidden_spin(first_spin, W_j, b_spin, 0)

    result.update()

    if normalize:
        result.normalize()

    return result


def add_hidden_spin(psi, position, connectivity, initial_value=(0.01 + 1j * math.pi / 4), noise=1e-6, hidden_spin_type=0):
    W_j = noise * complex_noise(connectivity)
    W_j[connectivity // 2] += initial_value
    # print([type(x) for x in (position - connectivity // 2, list(W_j), complex(noise * complex_noise(1)[0]))])
    psi.add_hidden_spin((position - connectivity // 2 + psi.N) % psi.N, W_j, noise * complex_noise(1)[0], hidden_spin_type)


def add_hidden_layer(psi, connectivity, initial_value=(0.01 + 1j * math.pi / 4), noise=1e-6, hidden_spin_type=0):
    for i in range(psi.N):
        add_hidden_spin(psi, i, connectivity, initial_value, noise, hidden_spin_type)


def fresh_neural_network(
    N,
    M,
    F=None,
    polarization="x",
    initial_value=(0.01 + 1j * math.pi / 4),
    initial_n=1.0,
    solver="gd",
    mc_configuration=None,
    gpu=False,
    noise=1e-6
):
    assert M % N == 0

    a = np.zeros(N, dtype=np.complex)
    b = np.zeros(M, dtype=np.complex)
    W = np.zeros((N, M), dtype=np.complex)
    n = np.zeros(M, dtype=np.complex)

    # for alpha in range(M // N):
    for alpha in range(1):
        W[:, alpha * N:(alpha + 1) * N] = initial_value * np.diag(np.ones(N))

    a += noise * ((2 * np.random.rand(N) - 1) + 1j * (2 * np.random.rand(N) - 1))
    b += noise * ((2 * np.random.rand(M) - 1) + 1j * (2 * np.random.rand(M) - 1))

    # for k in range(-2, 3):
    #     d = noise * ((2 * np.random.rand(N) - 1) + 1j * (2 * np.random.rand(N) - 1))
    #     W += np.diag(d[abs(k):], k=k)

    W += noise * ((2 * np.random.rand(N, M) - 1) + 1j * (2 * np.random.rand(N, M) - 1))

    iv, jv = np.meshgrid(np.arange(N), np.arange(N))
    for alpha in range(M // N):
        W[:, alpha * N: (alpha + 1) * N][np.minimum(abs(iv - jv), N - abs(iv - jv)) > 3] = 0

    n[:] = initial_n

    higher_order = F is not None
    if higher_order:
        assert F % N == 0

        X = np.zeros((N, F), dtype=np.complex)
        Y = np.zeros((F, M), dtype=np.complex)

        X[:, :N] = initial_value * np.diag(np.ones(N))
        X += noise**0.5 * (
            (2 * np.random.rand(N, F) - 1) + 1j * (2 * np.random.rand(N, F) - 1)
        )
        for alpha in range(1, 2):
            Y[:, alpha * N:(alpha + 1) * N] = np.diag(np.ones(N))

        # Y += noise**0.66 * (
        #     np.random.rand(F, M)**10 + 1j * np.random.rand(F, M)
        # )
        # Y[:, :N] = 0


    if polarization == "z":
        a[:] = 4

    prefactor = 1 / 2**((N + M) // 2)
    if higher_order:
        N_params = N + M + N * M + M + N * F + F * M
    else:
        N_params = N + M + N * M + M

    if mc_configuration is None:
        spin_ensemble = pyRBMonGPU.ExactSummation(N)
    else:
        spin_ensemble = pyRBMonGPU.MonteCarloLoop(
            mc_configuration.num_samples,
            mc_configuration.num_sweeps,
            mc_configuration.num_thermalization_sweeps,
            mc_configuration.num_markov_chains,
            gpu
        )

    mc = mc_configuration is not None

    if higher_order:
        NeuralNetwork_t = NeuralNetworkW3
        args = a, b, W, n, X, Y, prefactor, spin_ensemble, gpu, mc
    else:
        NeuralNetwork_t = NeuralNetwork
        args = a, b, W, n, prefactor, spin_ensemble, gpu, mc

    assert solver in [None, "gd", "sr"]
    if solver == "gd":
        solver = pyRBMonGPU.HilbertSpaceDistance(N_params, gpu)
        NeuralNetworkGD_t = type(NeuralNetworkGD.__name__, (NeuralNetwork_t,), dict(NeuralNetworkGD.__dict__))
        result = NeuralNetworkGD_t(solver, *args)
    elif solver == "sr":
        solver = pyRBMonGPU.DifferentiatePsi(N_params, gpu)
        NeuralNetworkSR_t = type(NeuralNetworkSR.__name__, (NeuralNetwork_t,), dict(NeuralNetworkSR.__dict__))
        result = NeuralNetworkSR_t(solver, *args)
    else:
        result = NeuralNetwork_t(*args)

    if not mc:
        result.prefactor /= result.norm

    return result
