import numpy as np
import matplotlib.pyplot as plt


def function_(x, y):
    return x ** 2 + y ** 2


def numerical_derivative_x(x, y, step):
    return (function_(x + step, y) - function_(x - step, y)) / (2 * step)


def numerical_derivative_y(x, y, step):
    return (function_(x, y + step) - function_(x, y - step)) / (2 * step)


def numerical_derivative_xx(x, y, step):
    return (numerical_derivative_x(x + step, y, step) - numerical_derivative_x(x - step, y, step)) / (2 * step)


def numerical_derivative_xy(x, y, step):
    return (numerical_derivative_x(x, y + step, step) - numerical_derivative_x(x, y - step, step)) / (2 * step)


def numerical_derivative_yy(x, y, step):
    return (numerical_derivative_y(x, y + step, step) - numerical_derivative_y(x, y - step, step)) / (2 * step)


def sgd(a, b, step):
    e = 1e-6
    gradient = [numerical_derivative_x(a, b, step), numerical_derivative_y(a, b, step)]
    iteration = 1
    error = []
    while np.abs(gradient[0]) > e or np.abs(gradient[1]) > e:
        a -= gradient[0] * step
        b -= gradient[1] * step
        gradient = [numerical_derivative_x(a, b, step), numerical_derivative_y(a, b, step)]
        iteration += 1
        error.append(np.sqrt(a ** 2 + b ** 2))
    error = [np.abs(x - np.sqrt(a ** 2 + b ** 2)) for x in error]
    return [a, b, function_(a, b), gradient[0], gradient[1], iteration, error]


def sgd_momentum(a, b, step):
    e = 1e-6
    gradient = [numerical_derivative_x(a, b, step), numerical_derivative_y(a, b, step)]
    u = [0, 0]
    mu = 0.9
    iteration = 1
    error = []
    while np.abs(gradient[0]) > e or np.abs(gradient[1]) > e:
        u[0] = mu * u[0] + (1 - mu) * gradient[0] * step
        u[1] = mu * u[1] + (1 - mu) * gradient[1] * step
        a -= u[0]
        b -= u[1]
        gradient = [numerical_derivative_x(a, b, step), numerical_derivative_y(a, b, step)]
        iteration += 1
        error.append(np.sqrt(a ** 2 + b ** 2))
    error = [np.abs(x - np.sqrt(a ** 2 + b ** 2)) for x in error]
    return [a, b, function_(a, b), gradient[0], gradient[1], iteration, error]


def sgd_nesterov_momentum(a, b, step):
    e = 1e-6
    gradient = [numerical_derivative_x(a, b, step), numerical_derivative_y(a, b, step)]
    u = [0, 0]
    mu = 0.9
    iteration = 1
    error = []
    while np.abs(gradient[0]) > e or np.abs(gradient[1]) > e:
        u[0] = mu * u[0] + (1 - mu) * gradient[0] * step
        u[1] = mu * u[1] + (1 - mu) * gradient[1] * step
        a -= u[0]
        b -= u[1]
        u_prev = u
        gradient = [numerical_derivative_x(a - mu * u_prev[0], b - mu * u_prev[1], step),
                    numerical_derivative_y(a - mu * u_prev[0], b - mu * u_prev[1], step)]
        iteration += 1
        error.append(np.sqrt(a ** 2 + b ** 2))
    error = [np.abs(x - np.sqrt(a ** 2 + b ** 2)) for x in error]
    return [a, b, function_(a, b), gradient[0], gradient[1], iteration, error]


def adagrad(a, b, step):
    e = 1e-6
    gradient = [numerical_derivative_x(a, b, step), numerical_derivative_y(a, b, step)]
    g = [gradient[0] ** 2, gradient[1] ** 2]
    iteration = 1
    error = []
    while np.abs(gradient[0]) > e or np.abs(gradient[1]) > e:
        a -= gradient[0] * step / np.sqrt(e + g[0])
        b -= gradient[1] * step / np.sqrt(e + g[1])
        gradient = [numerical_derivative_x(a, b, step), numerical_derivative_y(a, b, step)]
        g[0] += gradient[0] ** 2
        g[1] += gradient[1] ** 2
        iteration += 1
        error.append(np.sqrt(a ** 2 + b ** 2))
    error = [np.abs(x - np.sqrt(a ** 2 + b ** 2)) for x in error]
    return [a, b, function_(a, b), gradient[0], gradient[1], iteration, error]


def RMSProp(a, b, step):
    e = 1e-6
    gradient = [numerical_derivative_x(a, b, step), numerical_derivative_y(a, b, step)]
    g = [gradient[0] ** 2, gradient[1] ** 2]
    mu = 0.9
    iteration = 1
    error = []
    while np.abs(gradient[0]) > e or np.abs(gradient[1]) > e:
        a -= gradient[0] * step / np.sqrt(e + g[0])
        b -= gradient[1] * step / np.sqrt(e + g[1])
        gradient = [numerical_derivative_x(a, b, step), numerical_derivative_y(a, b, step)]
        g[0] = mu * g[0] + (1 - mu) * gradient[0] ** 2
        g[1] = mu * g[1] + (1 - mu) * gradient[1] ** 2
        iteration += 1
        error.append(np.sqrt(a ** 2 + b ** 2))
    error = [np.abs(x - np.sqrt(a ** 2 + b ** 2)) for x in error]
    return [a, b, function_(a, b), gradient[0], gradient[1], iteration, error]


def adadelta(a, b, step):
    e = 1e-6
    gradient = [numerical_derivative_x(a, b, step), numerical_derivative_y(a, b, step)]
    g = [gradient[0] ** 2, gradient[1] ** 2]
    deltag = [1, 1]
    mu = 0.9
    delta = [0, 0]
    iteration = 1
    error = []
    while np.abs(gradient[0]) > e or np.abs(gradient[1]) > e:
        delta[0] = -np.sqrt(e + deltag[0]) / np.sqrt(e + g[0]) * gradient[0]
        delta[1] = -np.sqrt(e + deltag[1]) / np.sqrt(e + g[1]) * gradient[1]
        a += delta[0] * step
        b += delta[1] * step
        gradient = [numerical_derivative_x(a, b, step), numerical_derivative_y(a, b, step)]
        deltag[0] = mu * deltag[0] + (1 - mu) * delta[0] ** 2
        deltag[1] = mu * deltag[1] + (1 - mu) * delta[1] ** 2
        g[0] = mu * g[0] + (1 - mu) * gradient[0] ** 2
        g[1] = mu * g[1] + (1 - mu) * gradient[1] ** 2
        iteration += 1
        error.append(np.sqrt(a ** 2 + b ** 2))
    error = [np.abs(x - np.sqrt(a ** 2 + b ** 2)) for x in error]
    return [a, b, function_(a, b), gradient[0], gradient[1], iteration, error]


def adam(a, b, step):
    e = 1e-6
    gradient = [numerical_derivative_x(a, b, step), numerical_derivative_y(a, b, step)]
    m = [0, 0]
    u = [0, 0]
    mu = 0.9
    mu2 = 0.999
    iteration = 1
    error = []
    while np.abs(gradient[0]) > e or np.abs(gradient[1]) > e:
        a -= m[0] / np.sqrt(u[0] + e) * step
        b -= m[1] / np.sqrt(u[1] + e) * step
        gradient = [numerical_derivative_x(a, b, step), numerical_derivative_y(a, b, step)]
        m[0] = mu * m[0] + (1 - mu) * gradient[0]
        m[1] = mu * m[1] + (1 - mu) * gradient[1]
        u[0] = mu2 * u[0] + (1 - mu2) * gradient[0] ** 2
        u[1] = mu2 * u[1] + (1 - mu2) * gradient[1] ** 2
        iteration += 1
        error.append(np.sqrt(a ** 2 + b ** 2))
    error = [np.abs(x - np.sqrt(a ** 2 + b ** 2)) for x in error]
    return [a, b, function_(a, b), gradient[0], gradient[1], iteration, error]


def newton(a, b, step):
    e = 1e-6
    gradient = [numerical_derivative_x(a, b, step), numerical_derivative_y(a, b, step)]
    iteration = 1
    error = []
    while np.abs(gradient[0]) > e or np.abs(gradient[1]) > e:
        dx2 = numerical_derivative_xx(a, b, step)
        dy2 = numerical_derivative_yy(a, b, step)
        dxdy = numerical_derivative_xy(a, b, step)
        determinant = dx2 * dy2 - dxdy ** 2
        a -= 1 / determinant * (dy2 * gradient[0] - dxdy * gradient[1])
        b -= 1 / determinant * (dx2 * gradient[1] - dxdy * gradient[0])
        gradient = [numerical_derivative_x(a, b, step), numerical_derivative_y(a, b, step)]
        iteration += 1
        error.append(np.sqrt(a ** 2 + b ** 2))
    error = [np.abs(x - np.sqrt(a ** 2 + b ** 2)) for x in error]
    return [a, b, function_(a, b), gradient[0], gradient[1], iteration, error]


def gauss_newton(a, b, step):
    e = 1e-6
    gradient = [numerical_derivative_x(a, b, step), numerical_derivative_y(a, b, step)]
    iteration = 1
    error = []
    while np.abs(gradient[0]) > e or np.abs(gradient[1]) > e:
        function_value = function_(a, b)
        a -= function_value * gradient[0] / (gradient[0] ** 2 + gradient[1] ** 2)
        b -= function_value * gradient[1] / (gradient[0] ** 2 + gradient[1] ** 2)
        gradient = [numerical_derivative_x(a, b, step), numerical_derivative_y(a, b, step)]
        iteration += 1
        error.append(np.sqrt(a ** 2 + b ** 2))
    error = [np.abs(x - np.sqrt(a ** 2 + b ** 2)) for x in error]
    return [a, b, function_(a, b), gradient[0], gradient[1], iteration, error]


def draw_function(infos):
    error_sgd = infos[0]
    error_momentum = infos[1]
    error_nesterov = infos[2]
    error_adagrad = infos[3]
    error_RMSProp = infos[4]
    error_adadelta = infos[5]
    error_adam = infos[6]
    error_newton = infos[7]
    error_gauss_newton = infos[7]
    values = []
    for i in range(1, 1001):
        values.append(i)
        if len(error_sgd) < i:
            error_sgd.append(error_sgd[-1])
        if len(error_momentum) < i:
            error_momentum.append(error_momentum[-1])
        if len(error_nesterov) < i:
            error_nesterov.append(error_nesterov[-1])
        if len(error_adagrad) < i:
            error_adagrad.append(error_adagrad[-1])
        if len(error_RMSProp) < i:
            error_RMSProp.append(error_RMSProp[-1])
        if len(error_adadelta) < i:
            error_adadelta.append(error_adadelta[-1])
        if len(error_adam) < i:
            error_adam.append(error_adam[-1])
        if len(error_newton) < i:
            error_newton.append(error_newton[-1])
        if len(error_gauss_newton) < i:
            error_gauss_newton.append(error_gauss_newton[-1])
    fig, ax = plt.subplots()
    ax.plot(values, error_sgd, label='sgd')
    ax.plot(values, error_momentum, label='momentum')
    ax.plot(values, error_nesterov, label='nesterov')
    ax.plot(values, error_adagrad, label='adagrad x200')
    ax.plot(values, error_RMSProp, label='RMSProp x30')
    ax.plot(values, error_adadelta, label='adadelta x100')
    ax.plot(values, error_adam, label='adam x10')
    ax.plot(values, error_newton, label='newton')
    ax.plot(values, error_gauss_newton, label='gauss-newton')
    ax.legend()
    ax.set(xlabel='iteration', ylabel='error', title='Error graph')
    ax.grid()

    plt.show()


if __name__ == '__main__':
    a_ = 10
    b_ = 20
    step_ = 0.01
    info_sgd = sgd(a_, b_, step_)
    print('sgd: x value = {0}, y value = {1}, function value = {2}, gradient = [{3}, {4}], iteration = {5}'
          .format(info_sgd[0], info_sgd[1], info_sgd[2], info_sgd[3], info_sgd[4], info_sgd[5]))
    info_momentum = sgd_momentum(a_, b_, step_)
    print('momentum: x value = {0}, y value = {1}, function value = {2}, gradient = [{3}, {4}], iteration = {5}'
          .format(info_momentum[0], info_momentum[1], info_momentum[2], info_momentum[3], info_momentum[4],
                  info_momentum[5]))
    info_nesterov = sgd_nesterov_momentum(a_, b_, step_)
    print('nesterov: x value = {0}, y value = {1}, function value = {2}, gradient = [{3}, {4}], iteration = {5}'
          .format(info_nesterov[0], info_nesterov[1], info_nesterov[2], info_nesterov[3], info_nesterov[4],
                  info_nesterov[5]))
    info_adagrad = adagrad(a_, b_, step_ * 200)
    print('adagrad: x value = {0}, y value = {1}, function value = {2}, gradient = [{3}, {4}], iteration = {5}'
          .format(info_adagrad[0], info_adagrad[1], info_adagrad[2], info_adagrad[3], info_adagrad[4], info_adagrad[5]))

    info_RMSProp = RMSProp(a_, b_, step_ * 30)
    print('RMSProp: x value = {0}, y value = {1}, function value = {2}, gradient = [{3}, {4}], iteration = {5}'
          .format(info_RMSProp[0], info_RMSProp[1], info_RMSProp[2], info_RMSProp[3], info_RMSProp[4], info_RMSProp[5]))
    info_adadelta = adadelta(a_, b_, step_ * 100)
    print('adadelta: x value = {0}, y value = {1}, function value = {2}, gradient = [{3}, {4}], iteration = {5}'
          .format(info_adadelta[0], info_adadelta[1], info_adadelta[2], info_adadelta[3], info_adadelta[4],
                  info_adadelta[5]))
    info_adam = adam(a_, b_, step_ * 10)
    print('adam: x value = {0}, y value = {1}, function value = {2}, gradient = [{3}, {4}], iteration = {5}'
          .format(info_adam[0], info_adam[1], info_adam[2], info_adam[3], info_adam[4], info_adam[5]))
    info_newton = newton(a_, b_, step_)
    print('newton: x value = {0}, y value = {1}, function value = {2}, gradient = [{3}, {4}], iteration = {5}'
          .format(info_newton[0], info_newton[1], info_newton[2], info_newton[3], info_newton[4], info_newton[5]))
    info_gauss_newton = gauss_newton(a_, b_, step_)
    print('gauss-newton: x value = {0}, y value = {1}, function value = {2}, gradient = [{3}, {4}], iteration = {5}'
          .format(info_gauss_newton[0], info_gauss_newton[1], info_gauss_newton[2], info_gauss_newton[3],
                  info_gauss_newton[4], info_gauss_newton[5]))
    draw_function([info_sgd[6], info_momentum[6], info_nesterov[6], info_adagrad[6], info_RMSProp[6], info_adadelta[6],
                   info_adam[6], info_newton[6], info_gauss_newton[6]])
