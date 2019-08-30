import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np

from math import sqrt


def plot_graph(X, Y, line):
    # create the graph
    fig, ax = plt.subplots()

    # plot points (x[i], y[i])
    ax.scatter(X, Y, c="black")

    # plot line
    ax.add_line(line)

    # set boundaries and plot
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.show()


def calc_grad_of_slope(X, Y, m, b):

    grad = 0

    for i in range(len(X)):
        grad += -2 * X[i] * (Y[i] - m * X[i] - b)

    return grad


def calc_grad_of_intercept(X, Y, m, b):

    grad = 0

    for i in range(len(X)):
        grad += -2 * (Y[i] - m * X[i] - b)

    return grad


def calc_grad_vec_magnitude(m_grad, b_grad):
    return sqrt(m_grad ** 2 + b_grad ** 2)


def calc_trend_y_intercept(X, Y):

    # calculate the mean of X and Y sets
    x_avg = np.mean(X)
    y_avg = np.mean(Y)

    # calculate the slope of line of best fit
    m = calc_trend_slope(X, Y)

    return y_avg - m * x_avg


def calc_trend_slope(X, Y):

    # calculate the covariance matrix
    cov_mat = np.cov(X, Y)

    # extract cov(X, Y) and var(X)
    cov_xy = cov_mat[0][1]
    var_x = cov_mat[0][0]

    return cov_xy / var_x


def get_linear_output(x, slope, y_int):
    return slope * x + y_int


if __name__ == "__main__":
    # set learning variables
    gamma = 0.001
    e = 0.0001
    model_slope = 3
    model_int = 10
    max_iter = 10000

    # dataset
    X = [5, 5.5, 8, 12, 17]
    Y = [1, 2, 4, 6, 7.5]
    x_max = max(X)

    # plot initial prediction
    model_line = lines.Line2D(
        [0, x_max],
        [model_int, get_linear_output(x_max, model_slope, model_int)],
        color="green"
    )
    plot_graph(X, Y, model_line)

    for i in range(max_iter):
        # calculate the derivatives for m and b
        m_grad = calc_grad_of_slope(X, Y,
                                    model_slope, model_int)
        b_grad = calc_grad_of_intercept(X, Y,
                                        model_slope, model_int)

        # calculate the gradient vector magnitude
        grad_mag = calc_grad_vec_magnitude(m_grad, b_grad)

        # print information for the current prediction
        print("{}) m: {} m_grad: {} b: {} b_grad: {} grad_mag: {}".format(
            i, model_slope, m_grad, model_int, b_grad, grad_mag
        ))

        # if the model is sufficiently fit, then exit loop
        if abs(grad_mag) < e:
            break

        # determine step size of variables
        slope_step_size = (-1) * gamma * m_grad
        int_step_size = (-1) * gamma * b_grad

        # recalculate the slope and y intercept
        model_slope = model_slope + slope_step_size
        model_int = model_int + int_step_size

    # create line
    model_line = lines.Line2D(
        [0, x_max],
        [model_int, get_linear_output(x_max, model_slope, model_int)],
        color="green"
    )
    plot_graph(X, Y, model_line)
