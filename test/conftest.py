from make_Psi import psi_random
from make_PsiDynamical import psi_random as psidynamical_random
# from pyRBMonGPU import ExactSummation, MonteCarloLoop
import quantum_tools as qt
from PauliExpression import PauliExpression


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", help="run tests also on GPU")


def pytest_generate_tests(metafunc):
    if 'gpu' in metafunc.fixturenames:
        if metafunc.config.getoption('gpu'):
            metafunc.parametrize("gpu", [True, False])
        else:
            metafunc.parametrize("gpu", [False])

    if 'mc' in metafunc.fixturenames:
        metafunc.parametrize("mc", [True, False])

    if 'psi' in metafunc.fixturenames:
        psi_list = [
            lambda gpu: psidynamical_random(2, 2, 0.01, gpu),
            # lambda gpu: psi_random(2, 2, 0.01, gpu),
            # lambda gpu: psi_random(5, 10, 0.1, gpu),
        ]
        metafunc.parametrize("psi", psi_list)

    if 'operator' in metafunc.fixturenames:
        operator_list = [
            lambda N, unitary: 0.05 * (
                PauliExpression(1.0j, {0: 2, 1: 2}) +
                PauliExpression(1.0j, {N - 2: 1, N - 1: 3})
            ) + (1 if unitary else 0),
            lambda N, unitary: 0.05j * qt.disordered_Heisenberg_chain(
                N, 1.0, 0.5, 2.0
            ) + (1 if unitary else 0)
        ]
        metafunc.parametrize("operator", operator_list)
