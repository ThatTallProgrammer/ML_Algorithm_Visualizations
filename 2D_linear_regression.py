"""
Example program for calculating the slope and
y-int for a line of best fit using gradient
descent

Joseph Cauthen
August 31, 2019
"""

import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np

from math import sqrt


def plot_graph(X, Y, line):
    """
    Plot and display a graph of model

    :param X: Known x values
    :param Y: Known f(x) values
    :param line: matplotlib.lines.Line2D object of model line
    :return: None
    """

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
    """
    Calculate the derivative of the loss
    function with respect to the slope

    :param X: List of known x values
    :param Y: List of known f(x) values
    :param m: Slope of the current model line
    :param b: y-int of the current model line
    :return: the derivative of the loss function
        with respect to m
    """
    grad = 0

    for i in range(len(X)):
        grad += -2 * X[i] * (Y[i] - m * X[i] - b)

    return grad


def calc_grad_of_intercept(X, Y, m, b):
    """
    Calculate the derivative of the loss
    function with respect to the slope

    :param X: List of known x values
    :param Y: List of known f(x) values
    :param m: Slope of the current model line
    :param b: y-int of the current model line
    :return: the derivative of the loss function
        with respect to m
    """

    grad = 0

    for i in range(len(X)):
        grad += -2 * (Y[i] - m * X[i] - b)

    return grad


def calc_grad_vec_magnitude(m_grad, b_grad):
    """
    Calculate the magnitude of the gradient
    vector [L'(slope), L'(y-int)]

    :param m_grad:
    :param b_grad:
    :return: the magnitude of the gradient vector
    """

    return sqrt(m_grad ** 2 + b_grad ** 2)


def calc_trend_y_intercept(X, Y):
    """
    Calculate the y-int for the line of best
    fit for the points (x, f(x)) using the
    statistical formula

    y_int = y_avg - (cov(x, f(x)) / var(x)) * x_avg

    :param X: List of known x values
    :param Y: List of known f(x) values
    :return: y-int of the line of best fit
    """

    # calculate the mean of X and Y sets
    x_avg = np.mean(X)
    y_avg = np.mean(Y)

    # calculate the slope of line of best fit
    m = calc_trend_slope(X, Y)


    return y_avg - m * x_avg


def calc_trend_slope(X, Y):
    """
    Calculate the slope of the line of best
    fit for points (x, f(x)) using the statistical
    formula

    slope = cov(x, f(x)) / var(x)

    :param X: List of known x values
    :param Y: List of known f(x) values
    :return: The slope of the line of best fit
    """

    # calculate the covariance matrix
    cov_mat = np.cov(X, Y)

    # extract cov(X, Y) and var(X)
    cov_xy = cov_mat[0][1]
    var_x = cov_mat[0][0]

    return cov_xy / var_x


def get_linear_output(x, slope, y_int):
    """
    Calculate the output for a linear function
    at x given the slope and the y-intercept

    f(x) = slope * x + y_int

    :param x: The value being evaulated
    :param slope: The slope of the line
    :param y_int: The y_int of the line
    :return: f(x)
    """

    return slope * x + y_int


if __name__ == "__main__":
    # set learning variables
    gamma = 0.001
    eps = 0.0001
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

    # train the model
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
        if abs(grad_mag) < eps:
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
