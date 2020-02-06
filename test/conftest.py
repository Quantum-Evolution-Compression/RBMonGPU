from pyRBMonGPU import new_neural_network, new_deep_neural_network, MonteCarloLoop, ExactSummation
import quantum_tools as qt


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
            lambda gpu: new_neural_network(2, 2, gpu=gpu),
            lambda gpu: new_neural_network(3, 9, noise=1e-2, gpu=gpu),
        ]
        metafunc.parametrize("psi", psi_list)

    if 'psi_deep' in metafunc.fixturenames:
        psi_list = [
            lambda gpu: new_deep_neural_network(2, [2], [2], gpu=gpu),
            # lambda gpu: new_deep_neural_network(3, [9, 6], [1, 3], noise=1e-2, gpu=gpu),
            # lambda gpu: new_deep_neural_network(8, [16, 8, 4], [4, 2, 4], noise=1e-2, gpu=gpu),
        ]
        metafunc.parametrize("psi_deep", psi_list)

    if 'psi_all' in metafunc.fixturenames:
        psi_list = [
            lambda gpu: new_neural_network(3, 9, noise=1e-2, gpu=gpu),
            lambda gpu: new_neural_network(4, 8, noise=1e-2, alpha=0.5, beta=0.5, free_quantum_axis=True, gpu=gpu),
            lambda gpu: new_neural_network(8, 24, noise=1e-2, alpha=1, beta=1, free_quantum_axis=True, gpu=gpu),
            lambda gpu: new_deep_neural_network(2, [2], [2], gpu=gpu),
            lambda gpu: new_deep_neural_network(8, [16, 8, 4], [4, 2, 4], noise=1e-2, gpu=gpu),
            lambda gpu: new_deep_neural_network(6, [12, 8, 4], [4, 3, 4], noise=1e-3, alpha=1, beta=1, free_quantum_axis=True, gpu=gpu),
        ]
        metafunc.parametrize("psi_all", psi_list)

    if 'hamiltonian' in metafunc.fixturenames:
        metafunc.parametrize(
            "hamiltonian",
            [
                lambda L: qt.disordered_Heisenberg_chain(L, 1, 0.2, 1)
            ]
        )

    if 'spin_ensemble' in metafunc.fixturenames:
        metafunc.parametrize(
            "spin_ensemble",
            [
                lambda L, gpu: ExactSummation(L, gpu)
            ]
        )
