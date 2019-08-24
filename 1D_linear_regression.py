import time

import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
import matplotlib.lines as lines


def plot_graph(x, y, trend_line, line):
    # create the graph
    fig, ax = plt.subplots()

    # plot X and Y points
    ax.scatter(X, Y, c='black')

    # add line of best fit
    ax.add_line(trend_line)

    # add additional line
    ax.add_line(line)

    # show the graph
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.show()


def calc_loss(x, y, slope, y_int):
    loss = 0

    for i in range(len(x)):
        loss += (y[i] - y_int - slope * x[i]) ** 2

    return loss / len(x)


def calc_gradient(x, y, slope, y_int):

    gradient = 0

    for i in range(len(x)):
        gradient += -2 * x[i] * (y[i] - y_int - slope * x[i])

    return gradient / len(x)


def calc_trend_slope(x, y):

    # calculate the covariance matrix
    cov_mat = np.cov(x, y)

    # get covariance of X, Y and variance of X
    cov = cov_mat[0][1]
    var_x = cov_mat[0][0]

    return cov / var_x


def calc_trend_y_intercept(trend_slope, x, y):

    # calculate the x and y averages
    x_avg = stats.mean(x)
    y_avg = stats.mean(y)

    return y_avg - trend_slope * x_avg


def get_1d_linear_output(x, m, b):
    return m * x + b


if __name__ == "__main__":
    X = [3, 5, 10, 20, 25]
    Y = [1, 3, 5, 7, 10]
    step_size = 0.002
    model_slope = 2

    # calculate line of best fit constants
    trend_slope = calc_trend_slope(X, Y)
    trend_y_int = calc_trend_y_intercept(trend_slope, X, Y)

    while True:
        gradient = calc_gradient(X, Y, model_slope, trend_y_int)
        loss = calc_loss(X, Y, model_slope, trend_y_int)

        print(
            "Slope: {} Y_Int: {} Loss: {} Gradient: {}".format(
                model_slope,
                trend_y_int,
                loss,
                gradient
            )
        )

        # generate line of best fit
        trend_line = lines.Line2D(
            [0, 25],
            [trend_y_int, get_1d_linear_output(25, trend_slope, trend_y_int)],
            color='red'
        )

        # generated model line
        model_line = lines.Line2D(
            [0, 25],
            [trend_y_int, get_1d_linear_output(25, model_slope, trend_y_int)],
            color='blue'
        )

        plot_graph(X, Y, trend_line, model_line)

        if abs(gradient) < 0.001:
            break

        step = (-1) * step_size * gradient
        model_slope = model_slope + step

        time.sleep(2)


