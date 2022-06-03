import numpy as np
import matplotlib.pyplot as plt

e = 1e-6


def function_(x, y):
    return x ** 2 + y ** 2


def residuals():
    return [lambda x, y: x, lambda x, y: y]


def numerical_derivative_x(x, y, func, step):
    return (func(x + step, y) - func(x - step, y)) / (2 * step)


def numerical_derivative_y(x, y, func, step):
    return (func(x, y + step) - func(x, y - step)) / (2 * step)


def numerical_derivative_xx(x, y, func, step):
    return (numerical_derivative_x(x + step, y, func, step) - numerical_derivative_x(x - step, y, func, step))\
           / (2 * step)


def numerical_derivative_xy(x, y, func, step):
    return (numerical_derivative_x(x, y + step, func, step) - numerical_derivative_x(x, y - step, func, step))\
           / (2 * step)


def numerical_derivative_yy(x, y, func, step):
    return (numerical_derivative_y(x, y + step, func, step) - numerical_derivative_y(x, y - step, func, step))\
           / (2 * step)


def grad_desc(a, b, step, learning_rate):
    gradient = [numerical_derivative_x(a, b, function_, step),
                numerical_derivative_y(a, b, function_, step)]
    iteration = 0
    error = []
    while np.abs(gradient[0]) > e or np.abs(gradient[1]) > e:
        a -= gradient[0] * learning_rate
        b -= gradient[1] * learning_rate
        gradient = [numerical_derivative_x(a, b, function_, step),
                    numerical_derivative_y(a, b, function_, step)]
        iteration += 1
        error.append(np.sqrt(a ** 2 + b ** 2))
    error = [np.abs(x - np.sqrt(a ** 2 + b ** 2)) for x in error]
    return [a, b, function_(a, b), gradient[0], gradient[1], iteration, error]


def momentum(a, b, step, learning_rate, decay_rate):
    gradient = [numerical_derivative_x(a, b, function_, step),
                numerical_derivative_y(a, b, function_, step)]
    u = [0, 0]
    iteration = 0
    error = []
    while np.abs(gradient[0]) > e or np.abs(gradient[1]) > e:
        u[0] = decay_rate * u[0] + (1 - decay_rate) * gradient[0] * learning_rate
        u[1] = decay_rate * u[1] + (1 - decay_rate) * gradient[1] * learning_rate
        a -= u[0]
        b -= u[1]
        gradient = [numerical_derivative_x(a, b, function_, step),
                    numerical_derivative_y(a, b, function_, step)]
        iteration += 1
        error.append(np.sqrt(a ** 2 + b ** 2))
    error = [np.abs(x - np.sqrt(a ** 2 + b ** 2)) for x in error]
    return [a, b, function_(a, b), gradient[0], gradient[1], iteration, error]


def nesterov_momentum(a, b, step, learning_rate, decay_rate):
    gradient = [numerical_derivative_x(a, b, function_, step),
                numerical_derivative_y(a, b, function_, step)]
    u = [0, 0]
    iteration = 0
    error = []
    while np.abs(gradient[0]) > e or np.abs(gradient[1]) > e:
        u[0] = decay_rate * u[0] + (1 - decay_rate) * gradient[0] * learning_rate
        u[1] = decay_rate * u[1] + (1 - decay_rate) * gradient[1] * learning_rate
        a -= u[0]
        b -= u[1]
        u_prev = u
        gradient = [numerical_derivative_x(a - decay_rate * u_prev[0], b - decay_rate * u_prev[1], function_, step),
                    numerical_derivative_y(a - decay_rate * u_prev[0], b - decay_rate * u_prev[1], function_, step)]
        iteration += 1
        error.append(np.sqrt(a ** 2 + b ** 2))
    error = [np.abs(x - np.sqrt(a ** 2 + b ** 2)) for x in error]
    return [a, b, function_(a, b), gradient[0], gradient[1], iteration, error]


def adagrad(a, b, step, learning_rate):
    gradient = [numerical_derivative_x(a, b, function_, step),
                numerical_derivative_y(a, b, function_, step)]
    g = [gradient[0] ** 2, gradient[1] ** 2]
    iteration = 0
    error = []
    while np.abs(gradient[0]) > e or np.abs(gradient[1]) > e:
        a -= gradient[0] * learning_rate / np.sqrt(e + g[0])
        b -= gradient[1] * learning_rate / np.sqrt(e + g[1])
        gradient = [numerical_derivative_x(a, b, function_, step),
                    numerical_derivative_y(a, b, function_, step)]
        g[0] += gradient[0] ** 2
        g[1] += gradient[1] ** 2
        iteration += 1
        error.append(np.sqrt(a ** 2 + b ** 2))
    error = [np.abs(x - np.sqrt(a ** 2 + b ** 2)) for x in error]
    return [a, b, function_(a, b), gradient[0], gradient[1], iteration, error]


def RMSProp(a, b, step, learning_rate, decay_rate):
    gradient = [numerical_derivative_x(a, b, function_, step),
                numerical_derivative_y(a, b, function_, step)]
    g = [gradient[0] ** 2, gradient[1] ** 2]
    iteration = 0
    error = []
    while np.abs(gradient[0]) > e or np.abs(gradient[1]) > e:
        a -= gradient[0] * learning_rate / np.sqrt(e + g[0])
        b -= gradient[1] * learning_rate / np.sqrt(e + g[1])
        gradient = [numerical_derivative_x(a, b, function_, step),
                    numerical_derivative_y(a, b, function_, step)]
        g[0] = decay_rate * g[0] + (1 - decay_rate) * gradient[0] ** 2
        g[1] = decay_rate * g[1] + (1 - decay_rate) * gradient[1] ** 2
        iteration += 1
        error.append(np.sqrt(a ** 2 + b ** 2))
    error = [np.abs(x - np.sqrt(a ** 2 + b ** 2)) for x in error]
    return [a, b, function_(a, b), gradient[0], gradient[1], iteration, error]


def adadelta(a, b, step, learning_rate, decay_rate):
    gradient = [numerical_derivative_x(a, b, function_, step),
                numerical_derivative_y(a, b, function_, step)]
    g = [gradient[0] ** 2, gradient[1] ** 2]
    deltag = [1, 1]
    delta = [0, 0]
    iteration = 0
    error = []
    while np.abs(gradient[0]) > e or np.abs(gradient[1]) > e:
        delta[0] = -np.sqrt(e + deltag[0]) / np.sqrt(e + g[0]) * gradient[0]
        delta[1] = -np.sqrt(e + deltag[1]) / np.sqrt(e + g[1]) * gradient[1]
        a += delta[0] * learning_rate
        b += delta[1] * learning_rate
        gradient = [numerical_derivative_x(a, b, function_, step),
                    numerical_derivative_y(a, b, function_, step)]
        deltag[0] = decay_rate * deltag[0] + (1 - decay_rate) * delta[0] ** 2
        deltag[1] = decay_rate * deltag[1] + (1 - decay_rate) * delta[1] ** 2
        g[0] = decay_rate * g[0] + (1 - decay_rate) * gradient[0] ** 2
        g[1] = decay_rate * g[1] + (1 - decay_rate) * gradient[1] ** 2
        iteration += 1
        error.append(np.sqrt(a ** 2 + b ** 2))
    error = [np.abs(x - np.sqrt(a ** 2 + b ** 2)) for x in error]
    return [a, b, function_(a, b), gradient[0], gradient[1], iteration, error]


def adam(a, b, step, learning_rate, decay_rate, beta):
    gradient = [numerical_derivative_x(a, b, function_, step),
                numerical_derivative_y(a, b, function_, step)]
    m = [0, 0]
    u = [0, 0]
    iteration = 0
    error = []
    while np.abs(gradient[0]) > e or np.abs(gradient[1]) > e:
        a -= m[0] / np.sqrt(u[0] + e) * learning_rate
        b -= m[1] / np.sqrt(u[1] + e) * learning_rate
        gradient = [numerical_derivative_x(a, b, function_, step),
                    numerical_derivative_y(a, b, function_, step)]
        m[0] = decay_rate * m[0] + (1 - decay_rate) * gradient[0]
        m[1] = decay_rate * m[1] + (1 - decay_rate) * gradient[1]
        u[0] = beta * u[0] + (1 - beta) * gradient[0] ** 2
        u[1] = beta * u[1] + (1 - beta) * gradient[1] ** 2
        iteration += 1
        error.append(np.sqrt(a ** 2 + b ** 2))
    error = [np.abs(x - np.sqrt(a ** 2 + b ** 2)) for x in error]
    return [a, b, function_(a, b), gradient[0], gradient[1], iteration, error]


def newton(a, b, step):
    gradient = [numerical_derivative_x(a, b, function_, step),
                numerical_derivative_y(a, b, function_, step)]
    iteration = 0
    error = []
    while np.abs(gradient[0]) > e or np.abs(gradient[1]) > e:
        dx2 = numerical_derivative_xx(a, b, function_, step)
        dy2 = numerical_derivative_yy(a, b, function_, step)
        dxdy = numerical_derivative_xy(a, b, function_, step)
        determinant = dx2 * dy2 - dxdy ** 2
        assert dx2 > 0 and determinant > 0, 'The hesse matrix is not positive definite'
        a -= 1 / determinant * (dy2 * gradient[0] - dxdy * gradient[1])
        b -= 1 / determinant * (dx2 * gradient[1] - dxdy * gradient[0])
        gradient = [numerical_derivative_x(a, b, function_, step),
                    numerical_derivative_y(a, b, function_, step)]
        iteration += 1
        error.append(np.sqrt(a ** 2 + b ** 2))
    error = [np.abs(x - np.sqrt(a ** 2 + b ** 2)) for x in error]
    return [a, b, function_(a, b), gradient[0], gradient[1], iteration, error]


def get_pseudo_inv_matrix(matrix):
    temp_matrix = []
    for col in range(len(matrix[0])):
        temp_matrix.append([])
        for col2 in range(len(matrix[0])):
            sum_ = 0
            for row in range(len(matrix)):
                sum_ += matrix[row][col] * matrix[row][col2]
            temp_matrix[col].append(sum_)
    determinant = temp_matrix[0][0] * temp_matrix[1][1] - temp_matrix[0][1] ** 2
    for row in range(2):
        for col in range(2):
            temp_matrix[row][col] /= determinant
    temp_matrix[0][1] *= -1
    temp_matrix[1][0] *= -1
    result_matrix = []
    for col in range(len(matrix)):
        result_matrix.append([])
        for col2 in range(len(temp_matrix[0])):
            sum_ = 0
            for row in range(len(matrix[0])):
                sum_ += matrix[col][row] * temp_matrix[row][col2]
            result_matrix[col].append(sum_)
    return result_matrix


def gauss_newton(a, b, step):
    gradient = [numerical_derivative_x(a, b, function_, step),
                numerical_derivative_y(a, b, function_, step)]
    iteration = 0
    error = []
    while np.abs(gradient[0]) > e or np.abs(gradient[1]) > e:
        residual_list = residuals()
        res_len = len(residual_list)
        jacob_matrix = []
        for i in range(res_len):
            jacob_matrix.append([numerical_derivative_x(a, b, residual_list[i], step),
                                 numerical_derivative_y(a, b, residual_list[i], step)])
        jacob_matrix = get_pseudo_inv_matrix(jacob_matrix)
        for i in range(res_len):
            residual_list[i] = residual_list[i](a, b)
        values = []
        for col in range(len(jacob_matrix)):
            sum_ = 0
            for row in range(len(jacob_matrix[0])):
                sum_ += jacob_matrix[col][row] * residual_list[col]
            values.append(sum_)
        a -= values[0]
        b -= values[1]
        gradient = [numerical_derivative_x(a, b, function_, step),
                    numerical_derivative_y(a, b, function_, step)]
        iteration += 1
        error.append(np.sqrt(a ** 2 + b ** 2))
    error = [np.abs(x - np.sqrt(a ** 2 + b ** 2)) for x in error]
    return [a, b, function_(a, b), gradient[0], gradient[1], iteration, error]


def draw_function(infos):
    max_values_count = 1000
    error_grad_desc = infos[0][:max_values_count]
    error_momentum = infos[1][:max_values_count]
    error_nesterov = infos[2][:max_values_count]
    error_adagrad = infos[3][:max_values_count]
    error_RMSProp = infos[4][:max_values_count]
    error_adadelta = infos[5][:max_values_count]
    error_adam = infos[6][:max_values_count]
    error_newton = infos[7][:max_values_count]
    error_gauss_newton = infos[8][:max_values_count]
    values = []
    for i in range(1, max_values_count + 1):
        values.append(i)
        if len(error_grad_desc) < i:
            error_grad_desc.append(error_grad_desc[-1])
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
    ax.plot(values, error_grad_desc, label='grad_desc')
    ax.plot(values, error_momentum, label='momentum')
    ax.plot(values, error_nesterov, label='nesterov')
    ax.plot(values, error_adagrad, label='adagrad x200')
    ax.plot(values, error_RMSProp, label='RMSProp x30')
    ax.plot(values, error_adadelta, label='adadelta x100')
    ax.plot(values, error_adam, label='adam x10')
    ax.plot(values, error_newton, label='newton')
    ax.plot(values, error_gauss_newton, label='gauss-newton')
    ax.set_xscale('log')
    ax.legend()
    ax.set(xlabel='iteration', ylabel='error', title='Error graph')
    ax.grid()

    plt.show()


if __name__ == '__main__':
    a_ = 10
    b_ = 20
    step_ = 0.001
    learning_rate_ = 0.01
    decay_rate_ = 0.9
    beta_adam = 0.999
    info_grad_desc = grad_desc(a_, b_, step_, learning_rate_)
    print('grad_desc: x value = {0}, y value = {1}, function value = {2}, gradient = [{3}, {4}], iteration = {5}'
          .format(info_grad_desc[0], info_grad_desc[1], info_grad_desc[2], info_grad_desc[3], info_grad_desc[4],
                  info_grad_desc[5]))
    info_momentum = momentum(a_, b_, step_, learning_rate_, decay_rate_)
    print('momentum: x value = {0}, y value = {1}, function value = {2}, gradient = [{3}, {4}], iteration = {5}'
          .format(info_momentum[0], info_momentum[1], info_momentum[2], info_momentum[3], info_momentum[4],
                  info_momentum[5]))
    info_nesterov = nesterov_momentum(a_, b_, step_, learning_rate_, decay_rate_)
    print('nesterov: x value = {0}, y value = {1}, function value = {2}, gradient = [{3}, {4}], iteration = {5}'
          .format(info_nesterov[0], info_nesterov[1], info_nesterov[2], info_nesterov[3], info_nesterov[4],
                  info_nesterov[5]))
    info_adagrad = adagrad(a_, b_, step_, learning_rate_ * 200)
    print('adagrad: x value = {0}, y value = {1}, function value = {2}, gradient = [{3}, {4}], iteration = {5}'
          .format(info_adagrad[0], info_adagrad[1], info_adagrad[2], info_adagrad[3], info_adagrad[4], info_adagrad[5]))
    info_RMSProp = RMSProp(a_, b_, step_, learning_rate_ * 30, decay_rate_)
    print('RMSProp: x value = {0}, y value = {1}, function value = {2}, gradient = [{3}, {4}], iteration = {5}'
          .format(info_RMSProp[0], info_RMSProp[1], info_RMSProp[2], info_RMSProp[3], info_RMSProp[4], info_RMSProp[5]))
    info_adadelta = adadelta(a_, b_, step_, learning_rate_ * 100, decay_rate_)
    print('adadelta: x value = {0}, y value = {1}, function value = {2}, gradient = [{3}, {4}], iteration = {5}'
          .format(info_adadelta[0], info_adadelta[1], info_adadelta[2], info_adadelta[3], info_adadelta[4],
                  info_adadelta[5]))
    info_adam = adam(a_, b_, step_, learning_rate_ * 10, decay_rate_, beta_adam)
    print('adam: x value = {0}, y value = {1}, function value = {2}, gradient = [{3}, {4}], iteration = {5}'
          .format(info_adam[0], info_adam[1], info_adam[2], info_adam[3], info_adam[4], info_adam[5]))
    info_newton = newton(a_, b_, step_)
    print('newton: x value = {0}, y value = {1}, function value = {2}, gradient = [{3}, {4}], iteration = {5}'
          .format(info_newton[0], info_newton[1], info_newton[2], info_newton[3], info_newton[4], info_newton[5]))
    info_gauss_newton = gauss_newton(a_, b_, step_)
    print('gauss-newton: x value = {0}, y value = {1}, function value = {2}, gradient = [{3}, {4}], iteration = {5}'
          .format(info_gauss_newton[0], info_gauss_newton[1], info_gauss_newton[2], info_gauss_newton[3],
                  info_gauss_newton[4], info_gauss_newton[5]))
    draw_function([info_grad_desc[6], info_momentum[6], info_nesterov[6], info_adagrad[6], info_RMSProp[6],
                   info_adadelta[6], info_adam[6], info_newton[6], info_gauss_newton[6]])
