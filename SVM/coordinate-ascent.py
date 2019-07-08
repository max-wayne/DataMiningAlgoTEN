# -*-coding:utf-8-*-
# @Created at: 2019-04-27 11:28
# @Author: Wayne
# Ref: https://blog.csdn.net/google19890102/article/details/51065297

import numpy as np
from matplotlib import pyplot as plt


def f(x1, x2):
    return -(x1**2)-3*(x2**2)+2*x1*x2+6


def main():
    # plot contour
    delta = 0.025
    x = np.arange(-3.0, 3.0, delta)
    y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = -(X**2)
    Z2 = -(Y**2)
    Z = 1.0*(Z1+3*Z2+2*X*Y)+6.0

    plt.figure()
    plt.contour(X, Y, Z)

    x1, x2 = 2, 2
    err = 1e-4
    x, y = [], []
    x.append(x1)
    y.append(x2)
    while True:
        f_0 = f(x1, x2)
        x1 = x2
        x.append(x1)
        y.append(x2)
        x2 = x1/3
        x.append(x1)
        y.append(x2)
        f_t = f(x1, x2)
        if abs(f_t-f_0) < err:
            break

    plt.plot(x[0], y[0], 'bs')
    plt.plot(x[1:-1], y[1:-1], 'k--')
    plt.plot(x[-1], y[-1], 'ro')

    plt.title('Coordinate Ascent')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


if __name__ == '__main__':
    main()




