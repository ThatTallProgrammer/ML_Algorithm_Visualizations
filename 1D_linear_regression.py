"""
Example program for calculating the slope
of a line of best fit using gradient descent

Joseph Cauthen
August 31, 2019
"""


import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
import matplotlib.lines as lines


def plot_graph(X, Y, line):
    """
    Plot a graph of the current model line

    :param X: The known x values
    :param Y: The knwon f(x) values
    :param line: The matplotlib.lines.Line2D object of the
        model line
    :return: None
    """

    # create the graph
    fig, ax = plt.subplots()

    # plot X and Y points
    ax.scatter(X, Y, c='black')

    # add line
    ax.add_line(line)

    # show the graph
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.show()


def calc_gradient(X, Y, slope, y_int):
    """
    Calculate the gradient as the L'(slope)

    :param X: The known x values
    :param Y: The known f(x) values
    :param slope: The current slope of the model line
    :param y_int: The current y-int of the model line
    :return: The derivative of the loss function at
        the current slope
    """

    grad = 0

    for i in range(len(X)):
        grad += -2 * X[i] * (Y[i] - y_int - slope * X[i])

    return grad / len(X)


def calc_trend_slope(X, Y):
    """
    Calculate the slope of the line of best
    fit for points (x, f(x)) using the statistical
    function

    slope = cov(x, y) / var(x)

    :param X: The known x values
    :param Y: The known f(x) values
    :return: The slope of the line of best fit
    """

    # calculate the covariance matrix
    cov_mat = np.cov(X, Y)

    # get covariance of X, Y and variance of X
    cov = cov_mat[0][1]
    var_x = cov_mat[0][0]

    return cov / var_x


def calc_trend_y_intercept(X, Y):
    """
    Calculate the y-int of the line of best
    fit using the statistical function

    y-int = y-avg - slope * x-avg

    :param X: List of known x values
    :param Y: List of known f(x) values
    :return: The y-int of the line of best fit
    """

    # calculate the x and y averages
    x_avg = stats.mean(X)
    y_avg = stats.mean(Y)

    # calculate the trend slope
    m = calc_trend_slope(X, Y)

    return y_avg - trend_slope * x_avg


def get_1d_linear_output(x, m, b):
    """
    Return the output of the linear
    function

    f(x) = m * x + b
    :param x: The x being evaluated
    :param m: The slope of the linear function
    :param b: The y-int of the linear function
    :return: f(x)
    """

    return m * x + b


if __name__ == "__main__":
    # specify the known data
    X = [3, 5, 10, 20, 25]
    Y = [1, 3, 5, 7, 10]

    # specify the training parameters
    gamma = 0.001
    eps = 0.0001
    model_slope = 10
    max_iter = 10000

    # calculate line of best fit constants
    trend_slope = calc_trend_slope(X, Y)
    trend_y_int = calc_trend_y_intercept(X, Y)

    # display model line before training
    x_max = max(X)
    model_line = lines.Line2D(
        [0, x_max],
        [trend_y_int, get_1d_linear_output(x_max, model_slope, trend_y_int)],
        color='blue'
    )
    plot_graph(X, Y, model_line)

    # train the model
    for i in range(max_iter):
        # calculate gradient for current model
        gradient = calc_gradient(X, Y, model_slope, trend_y_int)

        # update user with information
        print(
            "{}) Slope: {} Y_Int: {} Gradient: {}".format(
                i + 1,
                model_slope,
                trend_y_int,
                gradient
            )
        )

        # exit training if model is sufficiently accurate
        if abs(gradient) < eps:
            break

        # move opposite the direction of the gradient
        step = (-1) * gamma * gradient
        model_slope = model_slope + step

    # display the model line after training
    model_line = lines.Line2D(
        [0, x_max],
        [trend_y_int, get_1d_linear_output(x_max, model_slope, trend_y_int)],
        color='blue'
    )
    plot_graph(X, Y, model_line)
