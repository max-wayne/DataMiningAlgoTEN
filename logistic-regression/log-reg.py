# -*-coding:utf-8-*-
# @Created at: 2019-05-08 10:21
# @Author: Wayne
# Ref: https://github.com/max-wayne/AiLearning/blob/dev/blog/ml/5.Logistic回归.md
#   https://blog.csdn.net/moxigandashu/article/details/72779856
#   激活函数：https://blog.csdn.net/edogawachia/article/details/80043673
#           https://www.jianshu.com/p/22d9720dbf1a
#   经验之谈：https://www.cnblogs.com/ModifyRong/p/7739955.html
# Notes: w := w + alpha * grad|w * f(w)

import random
import numpy as np
from matplotlib import pyplot as plt


def gen_dataSet():
    """
    Generate dataSets.
    Returns:
        dataSet: 原始数据特征
        Label: 原始数据标签
    """
    rnd = np.random.RandomState(3)
    C1, L1 = [], np.zeros((20, 1))
    for i in range(20):
        x = rnd.uniform(-4, 1)
        y = rnd.uniform(4, 18)
        C1.append([1.0, x, y])  # 添加1.0构造常数项x0
    C1 = np.array(C1)

    C2, L2 = [], np.ones((20, 1))
    for i in range(20):
        x = rnd.uniform(-2, 4)
        y = rnd.uniform(-5, 10)
        C2.append([1.0, x, y])  # 添加1.0构造常数项x0
    C2 = np.array(C2)

    dataSet = np.concatenate((C1, C2))
    Label = np.concatenate((L1, L2))
    return dataSet, Label


def sigmoid(x):
    """
    Return 1.0 / (1 + exp(-x))
    Notes: Tanh是sigmoid的变形，前者是0均值，效果要好一些
            tanh(x) = 2*sigmoid(2x)-1
    """
    return 1.0 / (1 + np.exp(-x))


def gradAscent(dataSet, Label):
    """
    Optimize w.
    Args:
        dataSet: 原始数据集特征
        Label: 原始数据集标签
    Returns: w
    """
    dataSet = np.mat(dataSet)
    m, n = dataSet.shape
    # alpha: 每次移动步长 cycles:迭代次数
    alpha, cycles = 0.001, 500
    # 初始化每个特征的重为1
    weights = np.ones((n, 1))
    for i in range(cycles):
        h = sigmoid(dataSet*weights)
        error = Label - h
        weights += alpha * dataSet.T * error

    return weights


def stocGradAscent(dataSet, Label):
    """
    随机梯度上升算法：一次仅用一个样本点来更新回归系数
    Args:
        dataSet: 原始数据集特征
        Label: 原始数据集标签
    Returns: w
    Notes: gradAscent每次更新回归系数时都要遍历整个数据集，
            计算复杂度太高
    """
    m, n = dataSet.shape
    # alpha: 每次移动步长 cycles:迭代次数
    cycles = 500
    # 初始化每个特征的重为1
    weights = np.ones(n)
    for i in range(cycles):
        dataIndex = list(range(m))
        for j in range(m):
            alpha = 4/(1+i+j)+0.01  # 保证多次迭代后新数据依旧有贡献
            randIndex = int(random.uniform(0, len(dataIndex)))  # 减少周期波动
            h = sigmoid(np.dot(dataSet[randIndex], weights))
            error = Label[randIndex] - h
            weights += alpha * dataSet[randIndex] * error
            del dataIndex[randIndex]

    return weights


def plotBestFit(dataSet, Label, weights):
    """
    Plot dataSet and regression boundary.
    Args:
        dataSet: 原始数据集特征
        Label: 原始数据集标签
        weights: 回归系数
    Returns:
        None.
    """
    m = dataSet.shape[0]
    x1, y1 = list(), list()
    x2, y2 = list(), list()
    for i in range(m):
        if Label[i] == 0:
            x1.append(dataSet[i, 1])
            y1.append(dataSet[i, 2])
        else:
            x2.append(dataSet[i, 1])
            y2.append(dataSet[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1, y1, s=30, c='red')
    ax.scatter(x2, y2, s=30, c='green')

    x = np.arange(-4, 4, 0.1)
    print(weights)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y, color='k')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def main():
    dataSet, Label = gen_dataSet()
    # weights = gradAscent(dataSet, Label)
    weights = stocGradAscent(dataSet, Label)
    plotBestFit(dataSet, Label, weights)


if __name__ == '__main__':
    main()

