import numpy as np
from itertools import islice


def pack(array):
    N = len(array) // 2
    return array[:N] + 1j * array[N:]


def unpack(array):
    return np.hstack((array.real, array.imag))


def vanilla_generator(params, gradient_function, eta=1e-3):
    step = 0

    while True:
        gradient = gradient_function(step, params)
        params -= eta * gradient

        yield +params
        step += 1


def vanilla(params, num_steps, gradient_function, eta):
    history = []

    for step in range(num_steps):
        gradient = gradient_function(step, params)
        params -= eta * gradient

        history.append(+params)

    return history


def momentum(params, num_steps, gradient_function, eta, gamma=0.9):
    history = []
    velocity = 0

    for step in range(num_steps):
        gradient = gradient_function(step, params)
        velocity = gamma * velocity + eta * gradient
        params -= velocity

        history.append(+params)

    return history


def nag_generator(params, gradient_function, eta=1e-1, gamma=0.9, **kwargs):
    velocity = 0
    step = 0

    while True:
        gradient = gradient_function(step, params - gamma * velocity)
        velocity = gamma * velocity + eta * gradient
        params -= velocity

        yield +params
        step += 1


def nag(params, num_steps, gradient_function, eta=1e-1, gamma=0.9):
    gen = nag_generator(params, gradient_function, eta, gamma)

    return list(islice(gen, num_steps))


def adagrad(params, num_steps, gradient_function, eta, epsilon):
    history = []
    G = 0
    gradient = 0
    params_re = unpack(params)

    for step in range(num_steps):
        gradient = unpack(gradient_function(step, pack(params_re)))
        G += gradient**2

        params_re -= eta / (G + epsilon) * gradient
        history.append(pack(params_re))

    return history


def adadelta(params, num_steps, gradient_function, gamma, epsilon):
    history = []

    delta_params = 0
    E_delta_params2 = 0
    E_gradient2 = 0
    params_re = unpack(params)

    for step in range(num_steps):
        gradient = unpack(gradient_function(step, pack(params_re)))

        E_delta_params2 = gamma * E_delta_params2 + (1 - gamma) * delta_params**2
        E_gradient2 = gamma * E_gradient2 + (1 - gamma) * gradient**2

        rms_gradient = (E_gradient2 + epsilon)**0.5
        rms_delta_params = (E_delta_params2 + epsilon)**0.5

        delta_params = -rms_delta_params / rms_gradient * gradient

        params_re += delta_params
        history.append(pack(params_re))

    return history


def rmsprop_generator(params, gradient_function, eta, gamma=0.9, epsilon=1e-3, **kwargs):
    G = 0
    gradient = 0
    params_re = unpack(params)
    step = 0

    while True:
        gradient = unpack(gradient_function(step, pack(params_re)))
        G = gamma * G + (1 - gamma) * gradient**2

        eta_ = eta(step) if callable(eta) else eta

        params_re -= eta_ / (G + epsilon) * gradient

        yield pack(params_re)
        step += 1


def rmsprop(params, num_steps, gradient_function, eta, gamma, epsilon):
    gen = rmsprop_generator(params, gradient_function, eta, gamma, epsilon)

    return list(islice(gen, num_steps))


def adam_generator(params, gradient_function, beta1=0.9, beta2=0.999, eta=4e-3, epsilon=1e-3, **kwargs):
    m = 0
    v = 0
    params_re = unpack(params)
    step = 0

    while True:
        gradient = unpack(gradient_function(step, pack(params_re)))

        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient**2

        m_hat = m / (1 - beta1)
        v_hat = v / (1 - beta2)

        params_re -= eta / (v_hat**0.5 + epsilon) * m_hat
        yield pack(params_re)
        step += 1


def adam(params, num_steps, gradient_function, beta1=0.9, beta2=0.999, eta=4e-3, epsilon=1e-3):
    gen = adam_generator(params, gradient_function, beta1, beta2, eta, epsilon)

    return list(islice(gen, num_steps))


def padam_generator(params, gradient_function, beta1=0.9, beta2=0.999, eta=1e-1, epsilon=1e-4, p=1/4, **kwargs):
    m = 0
    v = 0
    v_hat = 0
    params_re = unpack(params)
    step = 0

    while True:
        gradient, cost = gradient_function(step, pack(params_re))
        gradient = unpack(gradient)

        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient**2

        v_hat = np.maximum(v_hat, v)

        eta_ = eta(step) if callable(eta) else eta

        params_re -= eta_ / (v_hat**p + epsilon) * m
        yield cost
        step += 1


def padam(params, num_steps, gradient_function, beta1=0.9, beta2=0.999, eta=1e-1, epsilon=1e-4, p=1/4):
    gen = padam_generator(params, gradient_function, beta1, beta2, eta, epsilon, p)

    return list(islice(gen, num_steps))


def adamax_generator(params, gradient_function, beta1=0.9, beta2=0.999, eta=1e-3, epsilon=1e-4, params_filter=None, **kwargs):
    m = 0
    u = 0
    gradient = 0
    params_re = unpack(params)
    step = 0

    while True:
        gradient = unpack(gradient_function(step, pack(params_re)))

        m = beta1 * m + (1 - beta1) * gradient
        m_hat = m / (1 - beta1)
        u = np.maximum(beta2 * u, abs(gradient))

        eta_ = eta(step) if callable(eta) else eta

        params_re -= eta_ / (u + epsilon) * m_hat

        if params_filter is not None:
            params_re = unpack(params_filter(
                step, pack(params_re)
            ))

        yield pack(params_re)
        step += 1


def adamax(params, num_steps, gradient_function, beta1=0.9, beta2=0.999, eta=1e-3, epsilon=1e-4, params_filter=None):
    gen = adamax_generator(params, gradient_function, beta1, beta2, eta, epsilon, params_filter)

    return list(islice(gen, num_steps))
