# -*-coding:utf-8-*-
# @Created at: 2019-04-26 23:11
# @Author: Wayne
# Ref: https://zhuanlan.zhihu.com/p/29212107
#       https://github.com/max-wayne/AiLearning/blob/dev/blog/ml/6.支持向量机.md

import random
import numpy as np
from matplotlib import pyplot as plt


def loadDataSet():
    dataSet, labels = [], []
    with open('./test.csv', 'r') as read_f:
        for row in read_f:
            x, y, label = [float(i) for i in row.strip('\n').split('\t')]
            dataSet.append([x, y])
            labels.append(label)

    return dataSet, labels


def select_j(i, m):
    # 在m中随机选择除了i之外剩余的数
    a = list(range(m))
    return random.choice(a[:i]+a[i+1:])


def clip(alpha, L, H):
    # 修剪alpha的值到L和H之间
    if alpha < L:
        return L
    elif alpha > H:
        return H
    else:
        return alpha


def get_w(alphas, dataSet, labels):
    # 通过已知数据点和拉格朗日乘子获得分割超平面参数
    alphas, dataSet, labels = np.array(alphas), np.array(dataSet), np.array(labels)
    yx = labels.reshape(1, -1).T * np.array([1, 1]) * dataSet
    w = np.dot(yx.T, alphas)
    return w.tolist()


def smoSimple(dataSet, labels, C, max_iter):
    """
    简化版SMO算法实现，未使用启发式方法对alpha对进行选择
    Args:
        dataSet: 所有特征数据向量
        labels: 所有数据标签
        C: 软间隔常数，0<=alpha_i<=C
        max_iter: 外层最大迭代次数
    Returns:
        alpha, b
    """
    dataSet = np.array(dataSet)
    labels = np.array(labels)
    m, n = dataSet.shape

    # 初始化参数
    alphas = np.zeros(m)
    b, iteration = 0, 0

    def f(x):
        # SVM分类器函数 y=w^Tx+b
        # Kernel function vector
        x = np.matrix(x).T
        data = np.matrix(dataSet)
        ks = data * x
        # predict value
        wx = np.matrix(alphas*labels)*ks
        fx = wx + b
        return fx[0, 0]

    while iteration < max_iter:
        pair_changed = 0
        for i in range(m):
            a_i, x_i, y_i = alphas[i], dataSet[i], labels[i]
            fx_i = f(x_i)
            E_i = fx_i - y_i
            j = select_j(i, m)
            a_j, x_j, y_j = alphas[j], dataSet[j], labels[j]
            fx_j = f(x_j)
            E_j = fx_j - y_j
            K_ii, K_jj, K_ij = np.dot(x_i, x_i), np.dot(x_j, x_j), np.dot(x_i, x_j)
            eta = K_ii + K_jj - 2*K_ij
            if eta <= 0:
                print('WARNING eta <= 0')
                continue
            # 获取更新的alpha对
            a_i_old, a_j_old = a_i, a_j
            a_j_new = a_j_old + y_j*(E_i-E_j)/eta
            # 对alpha进行修剪
            if y_i != y_j:
                L = max(0, a_j_old - a_i_old)
                H = min(C, C + a_j_old - a_i_old)
            else:
                L = max(0, a_i_old + a_j_old - C)
                H = min(C, a_j_old + a_i_old)
            a_j_new = clip(a_j_new, L, H)
            a_i_new = a_i_old + y_i*y_j*(a_j_old-a_j_new)
            if abs(a_j_new-a_j_old) < 1e-5:
                continue
            alphas[i], alphas[j] = a_i_new, a_j_new
            # 更新阈值b
            b_i = -E_i - y_i*K_ii*(a_i_new-a_i_old) - y_j*K_ij*(a_j_new-a_j_old) + b
            b_j = -E_j - y_i*K_ij*(a_i_new-a_i_old) - y_j*K_jj*(a_j_new-a_j_old) + b
            if 0 < a_i_new < C:
                b = b_i
            elif 0 < a_j_new < C:
                b = b_j
            else:
                b = (b_i+b_j) / 2
            pair_changed += 1
            print('INFO iteration:{} i:{} pair_changed:{}'.format(iteration, i, pair_changed))
        if pair_changed == 0:
            iteration += 1
        else:
            iteration = 0
        print('iteration:{}'.format(iteration))

    return alphas, b


def main():
    # load data
    dataSet, labels = loadDataSet()
    # 简化版SMO优化SVM算法
    alphas, b = smoSimple(dataSet, labels, 0.6, 40)
    # 分类数据点
    classified_pts = {'+1': [], '-1': []}
    for point, label in zip(dataSet, labels):
        if label == 1.0:
            classified_pts['+1'].append(point)
        else:
            classified_pts['-1'].append(point)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制数据点
    for label, pts in classified_pts.items():
        pts = np.array(pts)
        ax.scatter(pts[:, 0], pts[:, 1], label=label)
    # 绘制分割线
    w = get_w(alphas, dataSet, labels)
    x1, _ = max(dataSet, key=lambda x: x[0])
    x2, _ = min(dataSet, key=lambda x: x[0])
    a1, a2 = w
    y1, y2 = (-b-a1*x1)/a2, (-b*a1*x2)/a2
    ax.plot([x1, x2], [y1, y2])
    # 绘制支持向量
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 1e-3:
            x, y = dataSet[i]
            ax.scatter([x], [y], s=150, c='none', alpha=0.7,
                       linewidth=1.5, edgecolor='#AB3319')
    plt.show()


if __name__ == '__main__':
    main()


